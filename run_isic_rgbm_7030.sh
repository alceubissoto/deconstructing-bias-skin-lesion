CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv_rgbm.py with train_root=isic-rgbm train_csv=isic-csv/10fold/isic-train-0.csv --name sacred-isic-rgbm-7030-nodup-correct
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv_rgbm.py with train_root=isic-rgbm train_csv=isic-csv/10fold/isic-train-1.csv --name sacred-isic-rgbm-7030-nodup-correct
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv_rgbm.py with train_root=isic-rgbm train_csv=isic-csv/10fold/isic-train-2.csv --name sacred-isic-rgbm-7030-nodup-correct
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv_rgbm.py with train_root=isic-rgbm train_csv=isic-csv/10fold/isic-train-3.csv --name sacred-isic-rgbm-7030-nodup-correct
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv_rgbm.py with train_root=isic-rgbm train_csv=isic-csv/10fold/isic-train-4.csv --name sacred-isic-rgbm-7030-nodup-correct
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv_rgbm.py with train_root=isic-rgbm train_csv=isic-csv/10fold/isic-train-5.csv --name sacred-isic-rgbm-7030-nodup-correct
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv_rgbm.py with train_root=isic-rgbm train_csv=isic-csv/10fold/isic-train-6.csv --name sacred-isic-rgbm-7030-nodup-correct
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv_rgbm.py with train_root=isic-rgbm train_csv=isic-csv/10fold/isic-train-7.csv --name sacred-isic-rgbm-7030-nodup-correct
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv_rgbm.py with train_root=isic-rgbm train_csv=isic-csv/10fold/isic-train-8.csv --name sacred-isic-rgbm-7030-nodup-correct
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv_rgbm.py with train_root=isic-rgbm train_csv=isic-csv/10fold/isic-train-9.csv --name sacred-isic-rgbm-7030-nodup-correct

CUDA_VISIBLE_DEVICES=0 python3 test_csv_rgbm.py results-sacred/441/checkpoints/model_best.pth isic-rgbm/ isic-csv/10fold/isic-test-0.csv -n 50 -p > results-sacred/441/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv_rgbm.py results-sacred/442/checkpoints/model_best.pth isic-rgbm/ isic-csv/10fold/isic-test-1.csv -n 50 -p > results-sacred/442/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv_rgbm.py results-sacred/443/checkpoints/model_best.pth isic-rgbm/ isic-csv/10fold/isic-test-2.csv -n 50 -p > results-sacred/443/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv_rgbm.py results-sacred/444/checkpoints/model_best.pth isic-rgbm/ isic-csv/10fold/isic-test-3.csv -n 50 -p > results-sacred/444/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv_rgbm.py results-sacred/445/checkpoints/model_best.pth isic-rgbm/ isic-csv/10fold/isic-test-4.csv -n 50 -p > results-sacred/445/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv_rgbm.py results-sacred/446/checkpoints/model_best.pth isic-rgbm/ isic-csv/10fold/isic-test-5.csv -n 50 -p > results-sacred/446/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv_rgbm.py results-sacred/447/checkpoints/model_best.pth isic-rgbm/ isic-csv/10fold/isic-test-6.csv -n 50 -p > results-sacred/447/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv_rgbm.py results-sacred/448/checkpoints/model_best.pth isic-rgbm/ isic-csv/10fold/isic-test-7.csv -n 50 -p > results-sacred/448/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv_rgbm.py results-sacred/449/checkpoints/model_best.pth isic-rgbm/ isic-csv/10fold/isic-test-8.csv -n 50 -p > results-sacred/449/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv_rgbm.py results-sacred/450/checkpoints/model_best.pth isic-rgbm/ isic-csv/10fold/isic-test-9.csv -n 50 -p > results-sacred/450/auc_test_best_all.txt
