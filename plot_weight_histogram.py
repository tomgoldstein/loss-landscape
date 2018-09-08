import os
import numpy as np
import torch
import copy
import model_loader
import net_plotter
import math
import h5py
import matplotlib.pyplot as plt
import projection
import argparse

def plot_hist(w, w2, label1, label2, nbins, xmin, xmax, save_path):
    """
       Plot the weight distributions of two models.
    """
    fig, ax = plt.subplots()
    bins = np.linspace(xmin, xmax, nbins)
    ax.hist(d.numpy(), bins, alpha=0.5, label=label1)
    ax.hist(d2.numpy(), bins, alpha=0.5, label=label2)
    ax.legend(loc='upper right', fontsize='xx-large')
    plt.ylim(0, 5*math.pow(10,5))
    plt.tick_params('y', labelsize='xx-large')
    plt.tick_params('x', labelsize='xx-large')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(1,4))
    fig.savefig(save_path + '_histogram.pdf', dpi=200, bbox_inches='tight', format='pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot weight distribution')
    parser.add_argument('--model', default='vgg9', help='trained models')
    parser.add_argument('--save_folder', default='cifar10/trained_nets/histogram', help='path to save')
    parser.add_argument('--save_name', default='histogram_compare', help='filename to save')
    parser.add_argument('--file1', default='', help='path to the trained model file')
    parser.add_argument('--file2', default='', help='path to the trained model file2')
    parser.add_argument('--xmin', type=float, default=-0.01, help='min range value of weights')
    parser.add_argument('--xmax', type=float, default=0.01, help='max range value of weights')
    parser.add_argument('--nbins', default=100, help='number of bins')
    args = parser.parse_args()

    os.system('mkdir ' + args.save_folder)

    net = model_loader.load('cifar10', args.model, args.file1)
    net2 = model_loader.load('cifar10', args.model, args.file2)

    w = net_plotter.get_weights(net)
    w2 = net_plotter.get_weights(net2)

    d = projection.weights_to_vec(w)
    d2 = projection.weights_to_vec(w2)

    plot_hist(d, d2, 'bs=128', 'bs=8192', args.nbins, args.xmin, args.xmax,
              args.save_folder + '/' + args.save_name)
