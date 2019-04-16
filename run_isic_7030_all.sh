CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-rgb train_csv=isic-csv/10fold/isic-train-0.csv --name sacred-isic-rgb-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-rgb train_csv=isic-csv/10fold/isic-train-1.csv --name sacred-isic-rgb-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-rgb train_csv=isic-csv/10fold/isic-train-2.csv --name sacred-isic-rgb-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-rgb train_csv=isic-csv/10fold/isic-train-3.csv --name sacred-isic-rgb-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-rgb train_csv=isic-csv/10fold/isic-train-4.csv --name sacred-isic-rgb-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-rgb train_csv=isic-csv/10fold/isic-train-5.csv --name sacred-isic-rgb-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-rgb train_csv=isic-csv/10fold/isic-train-6.csv --name sacred-isic-rgb-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-rgb train_csv=isic-csv/10fold/isic-train-7.csv --name sacred-isic-rgb-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-rgb train_csv=isic-csv/10fold/isic-train-8.csv --name sacred-isic-rgb-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-rgb train_csv=isic-csv/10fold/isic-train-9.csv --name sacred-isic-rgb-7030-nodup

CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/361/checkpoints/model_best.pth isic-rgb/ isic-csv/10fold/isic-test-0.csv -n 50 -p > results-sacred/361/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/362/checkpoints/model_best.pth isic-rgb/ isic-csv/10fold/isic-test-1.csv -n 50 -p > results-sacred/362/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/363/checkpoints/model_best.pth isic-rgb/ isic-csv/10fold/isic-test-2.csv -n 50 -p > results-sacred/363/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/364/checkpoints/model_best.pth isic-rgb/ isic-csv/10fold/isic-test-3.csv -n 50 -p > results-sacred/364/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/365/checkpoints/model_best.pth isic-rgb/ isic-csv/10fold/isic-test-4.csv -n 50 -p > results-sacred/365/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/366/checkpoints/model_best.pth isic-rgb/ isic-csv/10fold/isic-test-5.csv -n 50 -p > results-sacred/366/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/367/checkpoints/model_best.pth isic-rgb/ isic-csv/10fold/isic-test-6.csv -n 50 -p > results-sacred/367/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/368/checkpoints/model_best.pth isic-rgb/ isic-csv/10fold/isic-test-7.csv -n 50 -p > results-sacred/368/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/369/checkpoints/model_best.pth isic-rgb/ isic-csv/10fold/isic-test-8.csv -n 50 -p > results-sacred/369/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/370/checkpoints/model_best.pth isic-rgb/ isic-csv/10fold/isic-test-9.csv -n 50 -p > results-sacred/370/auc_test_best_all.txt


# BACKGROUND

CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-background train_csv=isic-csv/10fold/isic-train-0.csv --name sacred-isic-background-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-background train_csv=isic-csv/10fold/isic-train-1.csv --name sacred-isic-background-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-background train_csv=isic-csv/10fold/isic-train-2.csv --name sacred-isic-background-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-background train_csv=isic-csv/10fold/isic-train-3.csv --name sacred-isic-background-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-background train_csv=isic-csv/10fold/isic-train-4.csv --name sacred-isic-background-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-background train_csv=isic-csv/10fold/isic-train-5.csv --name sacred-isic-background-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-background train_csv=isic-csv/10fold/isic-train-6.csv --name sacred-isic-background-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-background train_csv=isic-csv/10fold/isic-train-7.csv --name sacred-isic-background-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-background train_csv=isic-csv/10fold/isic-train-8.csv --name sacred-isic-background-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-background train_csv=isic-csv/10fold/isic-train-9.csv --name sacred-isic-background-7030-nodup

CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/371/checkpoints/model_best.pth isic-background/ isic-csv/10fold/isic-test-0.csv -n 50 -p > results-sacred/371/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/372/checkpoints/model_best.pth isic-background/ isic-csv/10fold/isic-test-1.csv -n 50 -p > results-sacred/372/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/373/checkpoints/model_best.pth isic-background/ isic-csv/10fold/isic-test-2.csv -n 50 -p > results-sacred/373/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/374/checkpoints/model_best.pth isic-background/ isic-csv/10fold/isic-test-3.csv -n 50 -p > results-sacred/374/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/375/checkpoints/model_best.pth isic-background/ isic-csv/10fold/isic-test-4.csv -n 50 -p > results-sacred/375/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/376/checkpoints/model_best.pth isic-background/ isic-csv/10fold/isic-test-5.csv -n 50 -p > results-sacred/376/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/377/checkpoints/model_best.pth isic-background/ isic-csv/10fold/isic-test-6.csv -n 50 -p > results-sacred/377/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/378/checkpoints/model_best.pth isic-background/ isic-csv/10fold/isic-test-7.csv -n 50 -p > results-sacred/378/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/379/checkpoints/model_best.pth isic-background/ isic-csv/10fold/isic-test-8.csv -n 50 -p > results-sacred/379/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/380/checkpoints/model_best.pth isic-background/ isic-csv/10fold/isic-test-9.csv -n 50 -p > results-sacred/380/auc_test_best_all.txt

# BBOX

CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox train_csv=isic-csv/10fold/isic-train-0.csv --name sacred-isic-bbox-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox train_csv=isic-csv/10fold/isic-train-1.csv --name sacred-isic-bbox-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox train_csv=isic-csv/10fold/isic-train-2.csv --name sacred-isic-bbox-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox train_csv=isic-csv/10fold/isic-train-3.csv --name sacred-isic-bbox-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox train_csv=isic-csv/10fold/isic-train-4.csv --name sacred-isic-bbox-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox train_csv=isic-csv/10fold/isic-train-5.csv --name sacred-isic-bbox-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox train_csv=isic-csv/10fold/isic-train-6.csv --name sacred-isic-bbox-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox train_csv=isic-csv/10fold/isic-train-7.csv --name sacred-isic-bbox-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox train_csv=isic-csv/10fold/isic-train-8.csv --name sacred-isic-bbox-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox train_csv=isic-csv/10fold/isic-train-9.csv --name sacred-isic-bbox-7030-nodup

CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/381/checkpoints/model_best.pth isic-bbox/ isic-csv/10fold/isic-test-0.csv -n 50 -p > results-sacred/381/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/382/checkpoints/model_best.pth isic-bbox/ isic-csv/10fold/isic-test-1.csv -n 50 -p > results-sacred/382/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/383/checkpoints/model_best.pth isic-bbox/ isic-csv/10fold/isic-test-2.csv -n 50 -p > results-sacred/383/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/384/checkpoints/model_best.pth isic-bbox/ isic-csv/10fold/isic-test-3.csv -n 50 -p > results-sacred/384/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/385/checkpoints/model_best.pth isic-bbox/ isic-csv/10fold/isic-test-4.csv -n 50 -p > results-sacred/385/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/386/checkpoints/model_best.pth isic-bbox/ isic-csv/10fold/isic-test-5.csv -n 50 -p > results-sacred/386/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/387/checkpoints/model_best.pth isic-bbox/ isic-csv/10fold/isic-test-6.csv -n 50 -p > results-sacred/387/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/388/checkpoints/model_best.pth isic-bbox/ isic-csv/10fold/isic-test-7.csv -n 50 -p > results-sacred/388/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/389/checkpoints/model_best.pth isic-bbox/ isic-csv/10fold/isic-test-8.csv -n 50 -p > results-sacred/389/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/390/checkpoints/model_best.pth isic-bbox/ isic-csv/10fold/isic-test-9.csv -n 50 -p > results-sacred/390/auc_test_best_all.txt

# BBOX 70

CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox-squared-70 train_csv=isic-csv/10fold/isic-train-0.csv --name sacred-isic-bbox-squared-70-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox-squared-70 train_csv=isic-csv/10fold/isic-train-1.csv --name sacred-isic-bbox-squared-70-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox-squared-70 train_csv=isic-csv/10fold/isic-train-2.csv --name sacred-isic-bbox-squared-70-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox-squared-70 train_csv=isic-csv/10fold/isic-train-3.csv --name sacred-isic-bbox-squared-70-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox-squared-70 train_csv=isic-csv/10fold/isic-train-4.csv --name sacred-isic-bbox-squared-70-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox-squared-70 train_csv=isic-csv/10fold/isic-train-5.csv --name sacred-isic-bbox-squared-70-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox-squared-70 train_csv=isic-csv/10fold/isic-train-6.csv --name sacred-isic-bbox-squared-70-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox-squared-70 train_csv=isic-csv/10fold/isic-train-7.csv --name sacred-isic-bbox-squared-70-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox-squared-70 train_csv=isic-csv/10fold/isic-train-8.csv --name sacred-isic-bbox-squared-70-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-bbox-squared-70 train_csv=isic-csv/10fold/isic-train-9.csv --name sacred-isic-bbox-squared-70-7030-nodup

CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/391/checkpoints/model_best.pth isic-bbox-squared-70/ isic-csv/10fold/isic-test-0.csv -n 50 -p > results-sacred/391/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/392/checkpoints/model_best.pth isic-bbox-squared-70/ isic-csv/10fold/isic-test-1.csv -n 50 -p > results-sacred/392/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/393/checkpoints/model_best.pth isic-bbox-squared-70/ isic-csv/10fold/isic-test-2.csv -n 50 -p > results-sacred/393/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/394/checkpoints/model_best.pth isic-bbox-squared-70/ isic-csv/10fold/isic-test-3.csv -n 50 -p > results-sacred/394/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/395/checkpoints/model_best.pth isic-bbox-squared-70/ isic-csv/10fold/isic-test-4.csv -n 50 -p > results-sacred/395/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/396/checkpoints/model_best.pth isic-bbox-squared-70/ isic-csv/10fold/isic-test-5.csv -n 50 -p > results-sacred/396/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/397/checkpoints/model_best.pth isic-bbox-squared-70/ isic-csv/10fold/isic-test-6.csv -n 50 -p > results-sacred/397/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/398/checkpoints/model_best.pth isic-bbox-squared-70/ isic-csv/10fold/isic-test-7.csv -n 50 -p > results-sacred/398/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/399/checkpoints/model_best.pth isic-bbox-squared-70/ isic-csv/10fold/isic-test-8.csv -n 50 -p > results-sacred/399/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/400/checkpoints/model_best.pth isic-bbox-squared-70/ isic-csv/10fold/isic-test-9.csv -n 50 -p > results-sacred/400/auc_test_best_all.txt


CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-label train_csv=isic-csv/10fold/isic-train-0.csv --name sacred-isic-label-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-label train_csv=isic-csv/10fold/isic-train-1.csv --name sacred-isic-label-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-label train_csv=isic-csv/10fold/isic-train-2.csv --name sacred-isic-label-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-label train_csv=isic-csv/10fold/isic-train-3.csv --name sacred-isic-label-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-label train_csv=isic-csv/10fold/isic-train-4.csv --name sacred-isic-label-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-label train_csv=isic-csv/10fold/isic-train-5.csv --name sacred-isic-label-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-label train_csv=isic-csv/10fold/isic-train-6.csv --name sacred-isic-label-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-label train_csv=isic-csv/10fold/isic-train-7.csv --name sacred-isic-label-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-label train_csv=isic-csv/10fold/isic-train-8.csv --name sacred-isic-label-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-label train_csv=isic-csv/10fold/isic-train-9.csv --name sacred-isic-label-7030-nodup

CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/401/checkpoints/model_best.pth isic-label/ isic-csv/10fold/isic-test-0.csv -n 50 -p > results-sacred/401/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/402/checkpoints/model_best.pth isic-label/ isic-csv/10fold/isic-test-1.csv -n 50 -p > results-sacred/402/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/403/checkpoints/model_best.pth isic-label/ isic-csv/10fold/isic-test-2.csv -n 50 -p > results-sacred/403/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/404/checkpoints/model_best.pth isic-label/ isic-csv/10fold/isic-test-3.csv -n 50 -p > results-sacred/404/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/405/checkpoints/model_best.pth isic-label/ isic-csv/10fold/isic-test-4.csv -n 50 -p > results-sacred/405/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/406/checkpoints/model_best.pth isic-label/ isic-csv/10fold/isic-test-5.csv -n 50 -p > results-sacred/406/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/407/checkpoints/model_best.pth isic-label/ isic-csv/10fold/isic-test-6.csv -n 50 -p > results-sacred/407/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/408/checkpoints/model_best.pth isic-label/ isic-csv/10fold/isic-test-7.csv -n 50 -p > results-sacred/408/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/409/checkpoints/model_best.pth isic-label/ isic-csv/10fold/isic-test-8.csv -n 50 -p > results-sacred/409/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/410/checkpoints/model_best.pth isic-label/ isic-csv/10fold/isic-test-9.csv -n 50 -p > results-sacred/410/auc_test_best_all.txt

CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-masked train_csv=isic-csv/10fold/isic-train-0.csv --name sacred-isic-masked-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-masked train_csv=isic-csv/10fold/isic-train-1.csv --name sacred-isic-masked-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-masked train_csv=isic-csv/10fold/isic-train-2.csv --name sacred-isic-masked-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-masked train_csv=isic-csv/10fold/isic-train-3.csv --name sacred-isic-masked-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-masked train_csv=isic-csv/10fold/isic-train-4.csv --name sacred-isic-masked-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-masked train_csv=isic-csv/10fold/isic-train-5.csv --name sacred-isic-masked-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-masked train_csv=isic-csv/10fold/isic-train-6.csv --name sacred-isic-masked-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-masked train_csv=isic-csv/10fold/isic-train-7.csv --name sacred-isic-masked-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-masked train_csv=isic-csv/10fold/isic-train-8.csv --name sacred-isic-masked-7030-nodup
CUDA_VISIBLE_DEVICES=0 python3 train_sacred_csv.py with train_root=isic-masked train_csv=isic-csv/10fold/isic-train-9.csv --name sacred-isic-masked-7030-nodup

CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/411/checkpoints/model_best.pth isic-masked/ isic-csv/10fold/isic-test-0.csv -n 50 -p > results-sacred/411/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/412/checkpoints/model_best.pth isic-masked/ isic-csv/10fold/isic-test-1.csv -n 50 -p > results-sacred/412/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/413/checkpoints/model_best.pth isic-masked/ isic-csv/10fold/isic-test-2.csv -n 50 -p > results-sacred/413/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/414/checkpoints/model_best.pth isic-masked/ isic-csv/10fold/isic-test-3.csv -n 50 -p > results-sacred/414/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/415/checkpoints/model_best.pth isic-masked/ isic-csv/10fold/isic-test-4.csv -n 50 -p > results-sacred/415/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/416/checkpoints/model_best.pth isic-masked/ isic-csv/10fold/isic-test-5.csv -n 50 -p > results-sacred/416/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/417/checkpoints/model_best.pth isic-masked/ isic-csv/10fold/isic-test-6.csv -n 50 -p > results-sacred/417/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/418/checkpoints/model_best.pth isic-masked/ isic-csv/10fold/isic-test-7.csv -n 50 -p > results-sacred/418/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/419/checkpoints/model_best.pth isic-masked/ isic-csv/10fold/isic-test-8.csv -n 50 -p > results-sacred/419/auc_test_best_all.txt
CUDA_VISIBLE_DEVICES=0 python3 test_csv.py results-sacred/420/checkpoints/model_best.pth isic-masked/ isic-csv/10fold/isic-test-9.csv -n 50 -p > results-sacred/420/auc_test_best_all.txt

