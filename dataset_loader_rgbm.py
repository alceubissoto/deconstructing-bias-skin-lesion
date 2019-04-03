import os
import os.path
import numpy as np
import pandas as pd
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
from torchvision.utils import save_image 
from tqdm import tqdm
from PIL import Image
import torchvision.transforms
import torchvision.transforms.functional as TF
import torch

# TODO: Make target_field optional for unannotated datasets.
class CSVDataset(data.Dataset):
    def __init__(self, root, csv_file, image_field, target_field,
                 loader=default_loader, transform=None,
                 target_transform=None, add_extension=None,
                 limit=None, random_subset_size=None,
                 split=None):
        self.root = root
        self.loader = loader
        self.image_field = image_field
        self.target_field = target_field
        self.transform = transform
        self.target_transform = target_transform
        self.add_extension = add_extension

        self.data = pd.read_csv(csv_file, sep=';')
        # Split
        if split is not None:
            with open(split, 'r') as f:
                selected_images = f.read().splitlines()
            self.data = self.data[self.data[image_field].isin(selected_images)]
            self.data = self.data.reset_index()

        # Calculate class weights for WeightedRandomSampler
        self.class_counts = dict(self.data['label'].value_counts())
        self.class_weights = {label: max(self.class_counts.values()) / count
                              for label, count in self.class_counts.items()}
        self.sampler_weights = [self.class_weights[cls]
                                for cls in self.data['label']]
        self.class_weights_list = [self.class_weights[k]
                                   for k in sorted(self.class_weights)]

        if random_subset_size:
            self.data = self.data.sample(n=random_subset_size)
            self.data = self.data.reset_index()

        if type(limit) == int:
            limit = (0, limit)
        if type(limit) == tuple:
            self.data = self.data[limit[0]:limit[1]]
            self.data = self.data.reset_index()

        classes = list(self.data[self.target_field].unique())
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes = classes

        print('Found {} images from {} classes.'.format(len(self.data),
                                                        len(classes)))
        for class_name, idx in self.class_to_idx.items():
            n_images = dict(self.data[self.target_field].value_counts())
            print("    Class '{}' ({}): {} images.".format(
                class_name, idx, n_images[class_name]))

    def __getitem__(self, index):
        path = os.path.join(self.root,
                            self.data.loc[index, self.image_field])
        if self.add_extension:
            path = path + self.add_extension
        sample = self.loader(path)
        samplergb = Image.fromarray(sample[:, :, :3])
        samplemask = Image.fromarray(sample[:, :, -1], mode='L')
        target = self.class_to_idx[self.data.loc[index, self.target_field]]
        
        #TRANSFORMS
        # Horizontal Flip
        p_hflip = np.random.binomial(size=1, n=1, p=0.5)[0]
        if p_hflip:
            samplergb = TF.hflip(samplergb)
            samplemask = TF.hflip(samplemask)
        # Vertical Flip
        p_vflip = np.random.binomial(size=1, n=1, p=0.5)[0]
        if p_vflip:
            samplergb = TF.vflip(samplergb)
            samplemask = TF.vflip(samplemask)
        # Random Resized Crop
        i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(samplergb, scale=(0.75, 1.0), ratio=(3. / 4., 4. / 3.))
        samplergb = TF.resized_crop(samplergb, i, j, h, w, (299, 299))
        samplemask = TF.resized_crop(samplemask, i, j, h, w, (299, 299))
        # Random Rotation
        angle = torchvision.transforms.RandomRotation.get_params((-45, 45))
        samplergb = TF.rotate(samplergb, angle)
        samplemask = TF.rotate(samplemask, angle)
        # ColorJitter
        hue = torchvision.transforms.ColorJitter(hue=0.2)
        samplergb = hue(samplergb)
        # ToTensor
        samplergb = TF.to_tensor(samplergb)
        samplemask = TF.to_tensor(samplemask)
        sample = torch.cat((samplergb, samplemask))
       
        sample = TF.normalize(sample, [0.485, 0.456, 0.406, 0.154], [0.229, 0.224, 0.225, 0.087])
        #sample = norm_transform(sample)
        #samplenp = sample.numpy()
        #samplergba = Image.fromarray(samplenp, 'RGBA')
        #samplergba.save('aug/PIL_' + str(index) + '.png') 
        #save_image(sample, 'aug/' + str(index) + '.png')
        return sample, target

    def __len__(self):
        return len(self.data)


class CSVDatasetWithName(CSVDataset):
    """
    CSVData that also returns image names.
    """

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        name = self.data.loc[i, self.image_field]
        return super().__getitem__(i), name
