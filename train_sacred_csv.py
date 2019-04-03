from itertools import islice
import os

import numpy as np
import pandas as pd
import pretrainedmodels as ptm
from sacred import Experiment
from sacred.observers import FileStorageObserver, TelegramObserver
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torchvision import models, datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from dataset_loader import CSVDatasetWithName

np.set_printoptions(precision=4, suppress=True)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


ex = Experiment()
fs_observer = FileStorageObserver.create('results-sacred')
ex.observers.append(fs_observer)
telegram_file = 'telegram.json'
if os.path.isfile(telegram_file):
    telegram_obs = TelegramObserver.from_config(telegram_file)
    ex.observers.append(telegram_obs)


@ex.config
def cfg():
    train_root = None
    train_csv = None
    val_root = None
    val_csv_low = None
    val_csv_medium = None
    val_csv_high = None
    n_classes = 2
    epochs = 60  # maximum number of epochs
    batch_size = 32  # batch size
    num_workers = 8  # parallel jobs for data loading and augmentation
    model_name = 'inceptionv4'  # model: inceptionv4, densenet161, resnet152, senet154
    val_samples = 8  # number of samples per image in validation
    early_stopping_patience = 22  # patience for early stopping
    weighted_loss = False  # use weighted loss based on class imbalance
    balanced_loader = False  # balance classes in data loader
    lr = 0.001  # base learning rate

def train_epoch(device, model, dataloaders, criterion, optimizer, phase,
                batches_per_epoch=None):
    losses = AverageMeter()
    accuracies = AverageMeter()
    all_preds = []
    all_labels = []
    if phase == 'train':
        model.train()
    else:
        model.eval()

    if batches_per_epoch:
        tqdm_loader = tqdm(
            islice(dataloaders['train'], 0, batches_per_epoch),
            total=batches_per_epoch)
    else:
        tqdm_loader = tqdm(dataloaders[phase])
    for data in tqdm_loader:
        (inputs, labels), name = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        if phase == 'train':
            optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        acc = torch.sum(preds == labels.data).item() / preds.shape[0]
        accuracies.update(acc)
        all_preds += list(F.softmax(outputs, dim=1).cpu().data.numpy())
        all_labels += list(labels.cpu().data.numpy())
        tqdm_loader.set_postfix(loss=losses.avg, acc=accuracies.avg)

    # Calculate multiclass AUC
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_preds[:, 1])

    # Confusion Matrix
    print('\nConfusion matrix')
    cm = confusion_matrix(all_labels, all_preds.argmax(axis=1))
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    print(cmn)
    acc = np.trace(cmn) / cmn.shape[0]

    return {'loss': losses.avg, 'acc': acc, 'auc': auc}


def save_images(dataset, to, n=32):
    for i in range(n):
        img_path = os.path.join(to, 'img_{}.png'.format(i))
        save_image(dataset[i][0], img_path)


@ex.automain
def main(train_root, train_csv, val_root, val_csv_low, val_csv_medium, val_csv_high, epochs, model_name, batch_size, 
         num_workers, val_samples, early_stopping_patience,
         n_classes, weighted_loss, balanced_loader, lr, _run):

    AUGMENTED_IMAGES_DIR = os.path.join(fs_observer.dir, 'images')
    CHECKPOINTS_DIR = os.path.join(fs_observer.dir, 'checkpoints')
    BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'model_best')
    LAST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'model_last')
    for directory in (AUGMENTED_IMAGES_DIR, CHECKPOINTS_DIR):
        os.makedirs(directory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ptm.inceptionv4(num_classes=1000, pretrained='imagenet')
    model.last_linear = nn.Linear(model.last_linear.in_features, n_classes)

    model.to(device)

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(299, scale=(0.75, 1.0)),
        transforms.RandomRotation(45),
        transforms.ColorJitter(hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(299, scale=(0.75, 1.0)),
        transforms.RandomRotation(45),
        transforms.ColorJitter(hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
        

    train_ds = CSVDatasetWithName(
        train_root, train_csv, 'image', 'label',
        transform=data_transforms['train'], add_extension='.png', split=None)


    datasets = {
        'train': train_ds,
    }


    if balanced_loader:
        data_sampler = sampler.WeightedRandomSampler(
            image_data['train'].sampler_weights, len(image_data['train']))
        shuffle = False
    else:
        data_sampler = None
        shuffle = True

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers,
                            sampler=data_sampler),
    }

    if weighted_loss:
        criterion = nn.CrossEntropyLoss(
            weight=torch.Tensor(image_data['train'].class_weights_list).cuda())
    else:
        criterion = nn.CrossEntropyLoss()


    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=0.001)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[25],
                                               gamma=0.1)

    metrics = {
        'train': pd.DataFrame(columns=['epoch', 'loss', 'acc', 'auc'])
    }

    best_val_auc_low = 0.0
    best_val_auc_medium = 0.0
    best_val_auc_high = 0.0
    epochs_without_improvement_low = 0
    epochs_without_improvement_medium = 0
    epochs_without_improvement_high = 0 
    batches_per_epoch = None

    for epoch in range(epochs):
        print('train epoch {}/{}'.format(epoch+1, epochs))
        epoch_train_result = train_epoch(
            device, model, dataloaders, criterion, optimizer, 'train',
            batches_per_epoch)

        metrics['train'] = metrics['train'].append(
            {**epoch_train_result, 'epoch': epoch}, ignore_index=True)
        print('train', epoch_train_result)
        
        scheduler.step()

    torch.save(model, BEST_MODEL_PATH+'.pth')
    for phase in ['train']:
        metrics[phase].epoch = metrics[phase].epoch.astype(int)
        metrics[phase].to_csv(os.path.join(fs_observer.dir, phase + '.csv'),
                              index=False)

    return {'max_train_acc': metrics['train']['acc'].max()} 
