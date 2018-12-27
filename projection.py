"""
    Project a model or multiple models to a plane spaned by given directions.
"""

import numpy as np
import torch
import os
import copy
import h5py
import net_plotter
import model_loader
import h5_util
from sklearn.decomposition import PCA

def tensorlist_to_tensor(weights):
    """ Concatnate a list of tensors into one tensor.

        Args:
            weights: a list of parameter tensors, e.g. net_plotter.get_weights(net).

        Returns:
            concatnated 1D tensor
    """
    return torch.cat([w.view(w.numel()) if w.dim() > 1 else torch.FloatTensor(w) for w in weights])


def nplist_to_tensor(nplist):
    """ Concatenate a list of numpy vectors into one tensor.

        Args:
            nplist: a list of numpy vectors, e.g., direction loaded from h5 file.

        Returns:
            concatnated 1D tensor
    """
    v = []
    for d in nplist:
        w = torch.tensor(d*np.float64(1.0))
        # Ignoreing the scalar values (w.dim() = 0).
        if w.dim() > 1:
            v.append(w.view(w.numel()))
        elif w.dim() == 1:
            v.append(w)
    return torch.cat(v)


def npvec_to_tensorlist(direction, params):
    """ Convert a numpy vector to a list of tensors with the same shape as "params".

        Args:
            direction: a list of numpy vectors, e.g., a direction loaded from h5 file.
            base: a list of parameter tensors from net

        Returns:
            a list of tensors with the same shape as base
    """
    if isinstance(params, list):
        w2 = copy.deepcopy(params)
        idx = 0
        for w in w2:
            w.copy_(torch.tensor(direction[idx:idx + w.numel()]).view(w.size()))
            idx += w.numel()
        assert(idx == len(direction))
        return w2
    else:
        s2 = []
        idx = 0
        for (k, w) in params.items():
            s2.append(torch.Tensor(direction[idx:idx + w.numel()]).view(w.size()))
            idx += w.numel()
        assert(idx == len(direction))
        return s2



def cal_angle(vec1, vec2):
    """ Calculate cosine similarities between two torch tensors or two ndarraies
        Args:
            vec1, vec2: two tensors or numpy ndarraies
    """
    if isinstance(vec1, torch.Tensor) and isinstance(vec1, torch.Tensor):
        return torch.dot(vec1, vec2)/(vec1.norm()*vec2.norm()).item()
    elif isinstance(vec1, np.ndarray) and isinstance(vec2, np.ndarray):
        return np.ndarray.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


def project_1D(w, d):
    """ Project vector w to vector d and get the length of the projection.

        Args:
            w: vectorized weights
            d: vectorized direction

        Returns:
            the projection scalar
    """
    assert len(w) == len(d), 'dimension does not match for w and '
    scale = torch.dot(w, d)/d.norm()
    return scale.item()


def project_2D(d, dx, dy, proj_method):
    """ Project vector d to the plane spanned by dx and dy.

        Args:
            d: vectorized weights
            dx: vectorized direction
            dy: vectorized direction
            proj_method: projection method
        Returns:
            x, y: the projection coordinates
    """

    if proj_method == 'cos':
        # when dx and dy are orthorgonal
        x = project_1D(d, dx)
        y = project_1D(d, dy)
    elif proj_method == 'lstsq':
        # solve the least squre problem: Ax = d
        A = np.vstack([dx.numpy(), dy.numpy()]).T
        [x, y] = np.linalg.lstsq(A, d.numpy())[0]

    return x, y


def project_trajectory(dir_file, w, s, dataset, model_name, model_files,
               dir_type='weights', proj_method='cos'):
    """
        Project the optimization trajectory onto the given two directions.

        Args:
          dir_file: the h5 file that contains the directions
          w: weights of the final model
          s: states of the final model
          model_name: the name of the model
          model_files: the checkpoint files
          dir_type: the type of the direction, weights or states
          proj_method: cosine projection

        Returns:
          proj_file: the projection filename
    """

    proj_file = dir_file + '_proj_' + proj_method + '.h5'
    if os.path.exists(proj_file):
        print('The projection file exists! No projection is performed unless %s is deleted' % proj_file)
        return proj_file

    # read directions and convert them to vectors
    directions = net_plotter.load_directions(dir_file)
    dx = nplist_to_tensor(directions[0])
    dy = nplist_to_tensor(directions[1])

    xcoord, ycoord = [], []
    for model_file in model_files:
        net2 = model_loader.load(dataset, model_name, model_file)
        if dir_type == 'weights':
            w2 = net_plotter.get_weights(net2)
            d = net_plotter.get_diff_weights(w, w2)
        elif dir_type == 'states':
            s2 = net2.state_dict()
            d = net_plotter.get_diff_states(s, s2)
        d = tensorlist_to_tensor(d)

        x, y = project_2D(d, dx, dy, proj_method)
        print ("%s  (%.4f, %.4f)" % (model_file, x, y))

        xcoord.append(x)
        ycoord.append(y)

    f = h5py.File(proj_file, 'w')
    f['proj_xcoord'] = np.array(xcoord)
    f['proj_ycoord'] = np.array(ycoord)
    f.close()

    return proj_file


def setup_PCA_directions(args, model_files, w, s):
    """
        Find PCA directions for the optimization path from the initial model
        to the final trained model.

        Returns:
            dir_name: the h5 file that stores the directions.
    """

    # Name the .h5 file that stores the PCA directions.
    folder_name = args.model_folder + '/PCA_' + args.dir_type
    if args.ignore:
        folder_name += '_ignore=' + args.ignore
    folder_name += '_save_epoch=' + str(args.save_epoch)
    os.system('mkdir ' + folder_name)
    dir_name = folder_name + '/directions.h5'

    # skip if the direction file exists
    if os.path.exists(dir_name):
        f = h5py.File(dir_name, 'a')
        if 'explained_variance_' in f.keys():
            f.close()
            return dir_name

    # load models and prepare the optimization path matrix
    matrix = []
    for model_file in model_files:
        print (model_file)
        net2 = model_loader.load(args.dataset, args.model, model_file)
        if args.dir_type == 'weights':
            w2 = net_plotter.get_weights(net2)
            d = net_plotter.get_diff_weights(w, w2)
        elif args.dir_type == 'states':
            s2 = net2.state_dict()
            d = net_plotter.get_diff_states(s, s2)
        if args.ignore == 'biasbn':
        	net_plotter.ignore_biasbn(d)
        d = tensorlist_to_tensor(d)
        matrix.append(d.numpy())

    # Perform PCA on the optimization path matrix
    print ("Perform PCA on the models")
    pca = PCA(n_components=2)
    pca.fit(np.array(matrix))
    pc1 = np.array(pca.components_[0])
    pc2 = np.array(pca.components_[1])
    print("angle between pc1 and pc2: %f" % cal_angle(pc1, pc2))

    print("pca.explained_variance_ratio_: %s" % str(pca.explained_variance_ratio_))

    # convert vectorized directions to the same shape as models to save in h5 file.
    if args.dir_type == 'weights':
        xdirection = npvec_to_tensorlist(pc1, w)
        ydirection = npvec_to_tensorlist(pc2, w)
    elif args.dir_type == 'states':
        xdirection = npvec_to_tensorlist(pc1, s)
        ydirection = npvec_to_tensorlist(pc2, s)

    if args.ignore == 'biasbn':
        net_plotter.ignore_biasbn(xdirection)
        net_plotter.ignore_biasbn(ydirection)

    f = h5py.File(dir_name, 'w')
    h5_util.write_list(f, 'xdirection', xdirection)
    h5_util.write_list(f, 'ydirection', ydirection)

    f['explained_variance_ratio_'] = pca.explained_variance_ratio_
    f['singular_values_'] = pca.singular_values_
    f['explained_variance_'] = pca.explained_variance_

    f.close()
    print ('PCA directions saved in: %s' % dir_name)

    return dir_name

