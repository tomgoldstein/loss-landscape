"""
    Calculate the hessian matrix of the projected surface and their eigen values.
"""

import argparse
import copy
import numpy as np
import h5py
import torch
import time
import socket
import os
import sys
import torchvision
import torch.nn as nn
import mpi4pytorch
import dataloader
import net_plotter
import plot_2D
import plot_1D
import model_loader
import scheduler
import projection as proj
import hess_vec_prod
from plot_surface import name_surface_file, setup_surface_file


def crunch_hessian_eigs(surf_file, net, w, s, d, dataloader, comm, rank, args):
    """
        Calculate eigen values of the hessian matrix of a given model in parallel
        using mpi reduce. This is the synchronized version.
    """
    f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')
    min_eig, max_eig = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if 'min_eig' not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        max_eig = -np.ones(shape=shape)
        min_eig = np.ones(shape=shape)
        if rank == 0:
            f['min_eig'] = min_eig
            f['max_eig'] = max_eig
    else:
        min_eig = f['min_eig'][:]
        max_eig = f['max_eig'][:]

    # Generate a list of all indices that need to be filled in.
    # The coordinates of each unfilled index are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(max_eig, xcoordinates, ycoordinates, comm)
    print('Computing %d values for rank %d'% (len(inds), rank))

    criterion = nn.CrossEntropyLoss() # set the loss function criteria

    # Loop over all un-calculated coords
    start_time = time.time()
    total_sync = 0.0

    for count, ind in enumerate(inds):
         # Get the coordinates of the points being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == 'weights':
            net_plotter.set_weights(net.module if args.ngpu > 1 else net, w, d, coord)
        elif args.dir_type == 'states':
            net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, coord)

        # Compute the eign values of the hessian matrix
        compute_start = time.time()
        maxeig, mineig, iter_count = hess_vec_prod.min_max_hessian_eigs(net, dataloader, \
                                        criterion, rank=rank, use_cuda=args.cuda, verbose=True)
        compute_time = time.time() - compute_start

        # Record the result in the local array
        max_eig.ravel()[ind] = maxeig
        min_eig.ravel()[ind] = mineig


        # Send updated plot data to the master node
        sync_start_time = time.time()
        max_eig = mpi4pytorch.reduce_max(comm, max_eig)
        min_eig = mpi4pytorch.reduce_min(comm, min_eig)
        sync_time = time.time() - sync_start_time
        total_sync += sync_time

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f['max_eig'][:] = max_eig
            f['min_eig'][:] = min_eig

        print("rank: %d %d/%d  (%0.2f%%)  %d\t  %s \tmaxeig:%8.5f \tmineig:%8.5f \titer: %d \ttime:%.2f \tsync:%.2f" % ( \
            rank, count + 1, len(inds), 100.0 * (count + 1)/len(inds), ind, str(coord), \
            maxeig, mineig, iter_count, compute_time, sync_time))

    # This is only needed to make MPI run smoothly. If this process has less work
    # than the rank0 process, then we need to keep calling allreduce so the rank0 process doesn't block
    for i in range(max(inds_nums) - len(inds)):
        max_eig = mpi4pytorch.reduce_max(comm, max_eig)
        min_eig = mpi4pytorch.reduce_min(comm, min_eig)

    total_time = time.time() - start_time
    print('Rank %d done! Total time: %f Sync: %f '%(rank, total_time, total_sync))
    f.close()


###############################################################
####                        MAIN
###############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use for each rank, useful for data parallel evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')

    # data parameters
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet')
    parser.add_argument('--datapath', default='cifar10/data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='no data preprocessing')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    # model parameters
    parser.add_argument('--model', default='resnet56', help='model name')
    parser.add_argument('--model_folder', default='', help='the common folder that contains model_file and model_file2')
    parser.add_argument('--model_file', default='', help='path to the trained model file')
    parser.add_argument('--model_file2', default='', help='use (model_file2 - model_file) as the xdirection')
    parser.add_argument('--model_file3', default='', help='use (model_file3 - model_file) as the ydirection')
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')

    # direction parameters
    parser.add_argument('--dir_file', default='', help='specify the name of direction file, or the path to an eisting direction file')
    parser.add_argument('--dir_type', default='weights', help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default=None, help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeatness experiment')
    parser.add_argument('--surf_file', default='', help='customize the name of surface file, could be an existing file.')

    # plot parameters
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')

    args = parser.parse_args()

    torch.manual_seed(123)
    #--------------------------------------------------------------------------
    # Environment setup
    #--------------------------------------------------------------------------
    if args.mpi:
        comm = mpi4pytorch.setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1

    # in case of multiple GPUs per node, set the GPU to use for each rank
    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception('User selected cuda option, but cuda is not available on this machine')
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        print('Rank %d use GPU %d of %d GPUs on %s' %
              (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))

    #--------------------------------------------------------------------------
    # Check plotting resolution
    #--------------------------------------------------------------------------
    try:
        args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
            assert args.ymin and args.ymax and args.ynum, \
            'You specified some arguments for the y axis, but not all'
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

    #--------------------------------------------------------------------------
    # Load models and extract parameters
    #--------------------------------------------------------------------------
    net = model_loader.load(args.dataset, args.model, args.model_file)
    w = net_plotter.get_weights(net) # initial parameters
    s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references
    if args.ngpu > 1:
        # data parallel with multiple GPUs on a single node
        net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    #--------------------------------------------------------------------------
    # Setup the direction file and the surface file
    #--------------------------------------------------------------------------
    dir_file = net_plotter.name_direction_file(args) # name the direction file
    if rank == 0:
        net_plotter.setup_direction(args, dir_file, net)

    surf_file = name_surface_file(args, dir_file)
    if rank == 0:
        setup_surface_file(args, surf_file, dir_file)

    # wait until master has setup the direction file and surface file
    mpi4pytorch.barrier(comm)

    # load directions
    d = net_plotter.load_directions(dir_file)
    # calculate the consine similarity of the two directions
    if len(d) == 2 and rank == 0:
        similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)

    #--------------------------------------------------------------------------
    # Setup dataloader
    #--------------------------------------------------------------------------
    # download CIFAR10 if it does not exit
    if rank == 0 and args.dataset == 'cifar10':
        torchvision.datasets.CIFAR10(root=args.dataset + '/data', train=True, download=True)

    mpi4pytorch.barrier(comm)

    trainloader, testloader = dataloader.load_dataset(args.dataset, args.datapath,
                                args.batch_size, args.threads, args.raw_data,
                                args.data_split, args.split_idx,
                                args.trainloader, args.testloader)

    #--------------------------------------------------------------------------
    # Start the computation
    #--------------------------------------------------------------------------
    crunch_hessian_eigs(surf_file, net, w, s, d, trainloader, comm, rank, args)
    print ("Rank " + str(rank) + ' is done!')

    #--------------------------------------------------------------------------
    # Plot figures
    #--------------------------------------------------------------------------
    if args.plot and rank == 0:
        if args.y:
            plot_2D.plot_2d_eig_ratio(surf_file, 'min_eig', 'max_eig', args.show)
        else:
            plot_1D.plot_1d_eig_ratio(surf_file, args.xmin, args.xmax, 'min_eig', 'max_eig')
