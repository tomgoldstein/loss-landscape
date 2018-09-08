# ===========================================================
# 2d loss contours for VGG-9
# ===========================================================

# ------------------------------------------------------------
# SGD
#------------------------------------------------------------
mpirun -n 4 python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model vgg9 \
--model_file cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=128_wd=0.0005_save_epoch=1/model_300.t7 \
--mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter

mpirun -n 4 python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model vgg9 \
--model_file cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=8192_wd=0.0005_save_epoch=1/model_300.t7 \
--mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter

mpirun -n 4 python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model vgg9 \
--model_file cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=128_wd=0.0_save_epoch=1/model_300.t7 \
--mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter

mpirun -n 4 python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model vgg9 \
--model_file cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=8192_wd=0.0_save_epoch=1/model_300.t7 \
--mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter

# ------------------------------------------------------------
# Adam
#------------------------------------------------------------
mpirun -n 4 python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model vgg9 \
--model_file cifar10/trained_nets/vgg9_adam_lr=0.001_bs=128_wd=0.0005_save_epoch=1/model_300.t7 \
--mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter

mpirun -n 4 python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model vgg9 \
--model_file cifar10/trained_nets/vgg9_adam_lr=0.001_bs=8192_wd=0.0005_save_epoch=1/model_300.t7 \
--mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter

mpirun -n 4 python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model vgg9 \
--model_file cifar10/trained_nets/vgg9_adam_lr=0.001_bs=128_wd=0.0_save_epoch=1/model_300.t7 \
--mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter

mpirun -n 4 python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model vgg9 \
--model_file cifar10/trained_nets/vgg9_adam_lr=0.001_bs=8192_wd=0.0_save_epoch=1/model_300.t7 \
--mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter
