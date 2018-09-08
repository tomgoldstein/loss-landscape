from matplotlib import pyplot as plt
import numpy as np
import torch
import argparse
import os
import cifar10
import net_plotter
import math
import h5py
import projection

def plot_l2_norm(model_name, model_folders, colors, labels, save_path, filename,
                 save_epoch=1, dir_type='weights'):
    epochs = range(0, 300 + save_epoch, save_epoch)
    if save_epoch != 1:
        epochs.append(1)
    epochs.sort()

    norms = []
    for model_folder in model_folders:
        norm = []
        for epoch in epochs:
            net = cifar10.model_loader.load(model_name, model_folder + '/model_' + str(epoch) + '.t7')
            w = net_plotter.get_weights(net)
            w_vec = projection.weights_to_vec(w)
            norm.append(torch.norm(w_vec))
        norms.append(norm)

    plt.figure()
    nepochs = len(norms[0])
    for i in range(0, len(model_folders)):
        plt.plot(np.arange(nepochs), norms[i], colors[i], label=labels[i], linewidth=1)
    plt.legend(loc=1, fontsize='xx-large')
    plt.tick_params('x', labelsize='x-large')
    plt.tick_params('y', labelsize='x-large')
    plt.xlabel('Epochs', fontsize='xx-large')
    plt.ylabel('L2-norm of weights', fontsize='xx-large')
    plt.savefig(save_path + '/' + filename + '_l2_norm_epoch.pdf', dpi=300, bbox_inches='tight', format='pdf')

    iters = [math.floor(50000/128) + 1, math.floor(50000/8192) + 1]
    plt.figure()
    for i in range(0, len(model_folders)):
        plt.plot(np.arange(0,nepochs*iters[i], iters[i]), norms[i], colors[i], label=labels[i], linewidth=1)
    plt.legend(loc=1,fontsize='xx-large')
    plt.tick_params('x', labelsize='x-large')
    plt.tick_params('y', labelsize='x-large')
    plt.xlabel('Iterations', fontsize='xx-large')
    plt.ylabel('L2-norm of weights', fontsize='xx-large')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,3))
    plt.savefig(save_path + '/' + filename + '_l2_norm_iteration.pdf', dpi=300, bbox_inches='tight', format='pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot weight norm')
    parser.add_argument('--model', default='vgg9', help='trained models')
    parser.add_argument('--save_folder', default='', help='folders for models to be projected')
    parser.add_argument('--dir_type', default='weights',
        help="""direction type: weights (all weights except bias and BN paras) |
                                states (include BN.running_mean/var)""")
    parser.add_argument('--model_folder1', default='', help='path to the trained model folder')
    parser.add_argument('--model_folder2', default='', help='path to the trained model folder')
    parser.add_argument('--label1', default='', help='legend label1')
    parser.add_argument('--label2', default='', help='legend label2')
    parser.add_argument('--save_file', default='', help='filename to save')
    args = parser.parse_args()

    os.system('mkdir ' + args.save_folder)
    model_folders = [args.model_folder1, args.model_folder2]
    colors = ['r', 'g']
    labels = [args.label1, args.label2]

    plot_l2_norm(args.model, model_folders, colors, labels, args.save_folder, args.save_file)
