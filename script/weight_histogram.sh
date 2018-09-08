# ==============================================================
# Ploting the weight distribution histogram
# ==============================================================
python plot_weight_histogram.py --model vgg9 \
--file1  cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=128_wd=0.0_save_epoch=1/model_300.t7 \
--file2  cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=8192_wd=0.0_save_epoch=1/model_300.t7 \
--xmin -0.2 --xmax 0.2 \
--save_folder cifar10/trained_nets/histogram \
--save_name vgg9_sgd_lr=0.1_wd=0.0_save_epoch=1

python plot_weight_histogram.py --model vgg9 \
--file1  cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=128_wd=0.0005_save_epoch=1/model_300.t7 \
--file2  cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=8192_wd=0.0005_save_epoch=1/model_300.t7 \
--xmin -0.01 --xmax 0.01 \
--save_folder cifar10/trained_nets/histogram \
--save_name vgg9_sgd_lr=0.1_wd=0.0005_save_epoch=1

python plot_weight_histogram.py --model vgg9 \
--file1  cifar10/trained_nets/vgg9_adam_lr=0.001_bs=128_wd=0.0_save_epoch=1/model_300.t7 \
--file2  cifar10/trained_nets/vgg9_adam_lr=0.001_bs=8192_wd=0.0_save_epoch=1/model_300.t7 \
--xmin -0.2 --xmax 0.2 \
--save_folder cifar10/trained_nets/histogram \
--save_name vgg9_adam_lr=0.001_wd=0.0_save_epoch=1

python plot_weight_histogram.py --model vgg9 \
--file1  cifar10/trained_nets/vgg9_adam_lr=0.001_bs=128_wd=0.0005_save_epoch=1/model_300.t7 \
--file2  cifar10/trained_nets/vgg9_adam_lr=0.001_bs=8192_wd=0.0005_save_epoch=1/model_300.t7 \
--xmin -0.05 --xmax 0.05 \
--save_folder cifar10/trained_nets/histogram \
--save_name vgg9_adam_lr=0.001_wd=0.0005_save_epoch=1
