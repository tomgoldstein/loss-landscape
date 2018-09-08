# ==============================================================
# Ploting the weight distribution histogram
# ==============================================================
python plot_weight_norm_curves.py --model vgg9 \
--model_folder1 cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=128_wd=0.0_save_epoch=1 \
--model_folder2 cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=8192_wd=0.0_save_epoch=1 \
--label1 bs=128 --label2 bs=8192 \
--save_folder cifar10/plots/weight_norm_curves \
--save_file vgg9_sgd_lr=0.1_wd=0.0_save_epoch=1


python plot_weight_norm_curves.py --model vgg9 \
--model_folder1 cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=128_wd=0.0005_save_epoch=1 \
--model_folder2 cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=8192_wd=0.0005_save_epoch=1 \
--label1 bs=128 --label2 bs=8192 \
--save_folder cifar10/plots/weight_norm_curves \
--save_file vgg9_sgd_lr=0.1_wd=0.0005_save_epoch=1


python plot_weight_norm_curves.py --model vgg9 \
--model_folder1 cifar10/trained_nets/vgg9_adam_lr=0.001_bs=128_wd=0.0_save_epoch=1 \
--model_folder2 cifar10/trained_nets/vgg9_adam_lr=0.001_bs=8192_wd=0.0_save_epoch=1 \
--label1 bs=128 --label2 bs=8192 \
--save_folder cifar10/plots/weight_norm_curves \
--save_file vgg9_adam_lr=0.001_wd=0.0_save_epoch=1


python plot_weight_norm_curves.py --model vgg9 \
--model_folder1 cifar10/trained_nets/vgg9_adam_lr=0.001_bs=128_wd=0.0005_save_epoch=1 \
--model_folder2 cifar10/trained_nets/vgg9_adam_lr=0.001_bs=8192_wd=0.0005_save_epoch=1 \
--label1 bs=128 --label2 bs=8192 \
--save_folder cifar10/plots/weight_norm_curves \
--save_file vgg9_adam_lr=0.001_wd=0.0005_save_epoch=1
