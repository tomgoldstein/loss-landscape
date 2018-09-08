# ===========================================================
# 1d normalized surface for ResNet-56-noshort
# ===========================================================

mpirun -n 4 python plot_surface.py --x=-1:1:51 --model resnet56_noshort \
--model_file cifar10/trained_nets/resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 \
--cuda --mpi --dir_type weights --xnorm filter --xignore biasbn
