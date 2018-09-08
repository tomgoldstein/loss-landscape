# ===========================================================
# 2d loss contours for ResNet-56-noshort
# ===========================================================

mpirun -n 4 python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model resnet56_noshort \
--model_file cifar10/trained_nets/resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 \
--mpi --cuda --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter

# ===========================================================
# 2d loss contours for ResNet-110-noshort
# ===========================================================
mpirun -n 4 python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model resnet110_noshort \
--model_file cifar10/trained_nets/resnet110_noshort_lr=0.01_bs=128/model_300.t7 --mpi --cuda \
--dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter
