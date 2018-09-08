# ===========================================================
# 1d linear interpolation for ResNet-56
# ===========================================================
mpirun -n 4 python plot_surface.py --cuda --mpi --x=-0.5:1.5:401 --model resnet56 --dir_type states \
--model_file cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0_save_epoch=1/model_300.t7 \
--model_file2 cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=4096_wd=0.0_save_epoch=1/model_300.t7

mpirun -n 4 python plot_surface.py --cuda --mpi --x=-0.5:1.5:401 --model resnet56 --dir_type states \
--model_file cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005_save_epoch=1/model_300.t7 \
--model_file2 cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=4096_wd=0.0005_save_epoch=1/model_300.t7

mpirun -n 4 python plot_surface.py --cuda --mpi --x=-0.5:1.5:401 --model resnet56 --dir_type states \
--model_file cifar10/trained_nets/resnet56_adam_lr=0.001_bs=128_wd=0.0_save_epoch=1/model_300.t7 \
--model_file2 cifar10/trained_nets/resnet56_adam_lr=0.001_bs=4096_wd=0.0_save_epoch=1/model_300.t7

mpirun -n 4 python plot_surface.py --cuda --mpi --x=-0.5:1.5:401 --model resnet56 --dir_type states \
--model_file cifar10/trained_nets/resnet56_adam_lr=0.001_bs=128_wd=0.0005_save_epoch=1/model_300.t7 \
--model_file2 cifar10/trained_nets/resnet56_adam_lr=0.001_bs=4096_wd=0.0005_save_epoch=1/model_300.t7
