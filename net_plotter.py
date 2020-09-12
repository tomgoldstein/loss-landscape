"""
    Manipulate network parameters and setup random directions with normalization.
"""

import torch
import copy
from os.path import exists, commonprefix
import h5py
import h5_util
import model_loader
import os

################################################################################
#                 Supporting functions for weights manipulation
################################################################################
def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]


def set_weights(net, weights, directions=None, step=None):
    """
        Overwrite the network's weights with a specified list of tensors
        or change weights along directions with a step size.
    """
    if directions is None:
        # You cannot specify a step length without a direction.
        for (p, w) in zip(net.parameters(), weights):
            p.data.copy_(w.type(type(p.data)))
    else:
        assert step is not None, 'If a direction is specified then step must be specified as well'

        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d*step for d in directions[0]]

        for (p, w, d) in zip(net.parameters(), weights, changes):
            p.data = w + torch.Tensor(d).type(type(w))


def set_states(net, states, directions=None, step=None):
    """
        Overwrite the network's state_dict or change it along directions with a step size.
    """
    if directions is None:
        net.load_state_dict(states)
    else:
        assert step is not None, 'If direction is provided then the step must be specified as well'
        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d*step for d in directions[0]]

        new_states = copy.deepcopy(states)
        assert (len(new_states) == len(changes))
        for (k, v), d in zip(new_states.items(), changes):
            d = torch.tensor(d)
            v.add_(d.type(v.type()))

        net.load_state_dict(new_states)


def get_random_weights(weights):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size()) for w in weights]


def get_random_states(states):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's state_dict(), so one direction entry
        per weight, including BN's running_mean/var.
    """
    return [torch.randn(w.size()) for k, w in states.items()]


def get_diff_weights(weights, weights2):
    """ Produce a direction from 'weights' to 'weights2'."""
    return [w2 - w for (w, w2) in zip(weights, weights2)]


def get_diff_states(states, states2):
    """ Produce a direction from 'states' to 'states2'."""
    return [v2 - v for (k, v), (k2, v2) in zip(states.items(), states2.items())]


################################################################################
#                        Normalization Functions
################################################################################
def normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.

        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    """
        The normalization scales the direction entries according to the entries of weights.
    """
    assert(len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


def normalize_directions_for_states(direction, states, norm='filter', ignore='ignore'):
    assert(len(direction) == len(states))
    for d, (k, w) in zip(direction, states.items()):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


def ignore_biasbn(directions):
    """ Set bias and bn parameters in directions to zero """
    for d in directions:
        if d.dim() <= 1:
            d.fill_(0)

################################################################################
#                       Create directions
################################################################################
def create_target_direction(net, net2, dir_type='states'):
    """
        Setup a target direction from one model to the other

        Args:
          net: the source model
          net2: the target model with the same architecture as net.
          dir_type: 'weights' or 'states', type of directions.

        Returns:
          direction: the target direction from net to net2 with the same dimension
                     as weights or states.
    """

    assert (net2 is not None)
    # direction between net2 and net
    if dir_type == 'weights':
        w = get_weights(net)
        w2 = get_weights(net2)
        direction = get_diff_weights(w, w2)
    elif dir_type == 'states':
        s = net.state_dict()
        s2 = net2.state_dict()
        direction = get_diff_states(s, s2)

    return direction


def create_random_direction(net, dir_type='weights', ignore='biasbn', norm='filter'):
    """
        Setup a random (normalized) direction with the same dimension as
        the weights or states.

        Args:
          net: the given trained model
          dir_type: 'weights' or 'states', type of directions.
          ignore: 'biasbn', ignore biases and BN parameters.
          norm: direction normalization method, including
                'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'

        Returns:
          direction: a random direction with the same dimension as weights or states.
    """

    # random direction
    if dir_type == 'weights':
        weights = get_weights(net) # a list of parameters.
        direction = get_random_weights(weights)
        normalize_directions_for_weights(direction, weights, norm, ignore)
    elif dir_type == 'states':
        states = net.state_dict() # a dict of parameters, including BN's running mean/var.
        direction = get_random_states(states)
        normalize_directions_for_states(direction, states, norm, ignore)

    return direction


def setup_direction(args, dir_file, net):
    """
        Setup the h5 file to store the directions.
        - xdirection, ydirection: The pertubation direction added to the mdoel.
          The direction is a list of tensors.
    """
    print('-------------------------------------------------------------------')
    print('setup_direction')
    print('-------------------------------------------------------------------')
    
    # Setup env for preventing lock on h5py file for newer h5py versions
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    # Skip if the direction file already exists
    if exists(dir_file):
        f = h5py.File(dir_file, 'r')
        if (args.y and 'ydirection' in f.keys()) or 'xdirection' in f.keys():
            f.close()
            print ("%s is already setted up" % dir_file)
            return
        f.close()

    # Create the plotting directions
    f = h5py.File(dir_file,'w') # create file, fail if exists
    if not args.dir_file:
        print("Setting up the plotting directions...")
        if args.model_file2:
            net2 = model_loader.load(args.dataset, args.model, args.model_file2)
            xdirection = create_target_direction(net, net2, args.dir_type)
        else:
            xdirection = create_random_direction(net, args.dir_type, args.xignore, args.xnorm)
        h5_util.write_list(f, 'xdirection', xdirection)

        if args.y:
            if args.same_dir:
                ydirection = xdirection
            elif args.model_file3:
                net3 = model_loader.load(args.dataset, args.model, args.model_file3)
                ydirection = create_target_direction(net, net3, args.dir_type)
            else:
                ydirection = create_random_direction(net, args.dir_type, args.yignore, args.ynorm)
            h5_util.write_list(f, 'ydirection', ydirection)

    f.close()
    print ("direction file created: %s" % dir_file)


def name_direction_file(args):
    """ Name the direction file that stores the random directions. """

    if args.dir_file:
        assert exists(args.dir_file), "%s does not exist!" % args.dir_file
        return args.dir_file

    dir_file = ""

    file1, file2, file3 = args.model_file, args.model_file2, args.model_file3

    # name for xdirection
    if file2:
        # 1D linear interpolation between two models
        assert exists(file2), file2 + " does not exist!"
        if file1[:file1.rfind('/')] == file2[:file2.rfind('/')]:
            # model_file and model_file2 are under the same folder
            dir_file += file1 + '_' + file2[file2.rfind('/')+1:]
        else:
            # model_file and model_file2 are under different folders
            prefix = commonprefix([file1, file2])
            prefix = prefix[0:prefix.rfind('/')]
            dir_file += file1[:file1.rfind('/')] + '_' + file1[file1.rfind('/')+1:] + '_' + \
                       file2[len(prefix)+1: file2.rfind('/')] + '_' + file2[file2.rfind('/')+1:]
    else:
        dir_file += file1

    dir_file += '_' + args.dir_type
    if args.xignore:
        dir_file += '_xignore=' + args.xignore
    if args.xnorm:
        dir_file += '_xnorm=' + args.xnorm

    # name for ydirection
    if args.y:
        if file3:
            assert exists(file3), "%s does not exist!" % file3
            if file1[:file1.rfind('/')] == file3[:file3.rfind('/')]:
               dir_file += file3
            else:
               # model_file and model_file3 are under different folders
               dir_file += file3[:file3.rfind('/')] + '_' + file3[file3.rfind('/')+1:]
        else:
            if args.yignore:
                dir_file += '_yignore=' + args.yignore
            if args.ynorm:
                dir_file += '_ynorm=' + args.ynorm
            if args.same_dir: # ydirection is the same as xdirection
                dir_file += '_same_dir'

    # index number
    if args.idx > 0: dir_file += '_idx=' + str(args.idx)

    dir_file += ".h5"

    return dir_file


def load_directions(dir_file):
    """ Load direction(s) from the direction file."""

    f = h5py.File(dir_file, 'r')
    if 'ydirection' in f.keys():  # If this is a 2D plot
        xdirection = h5_util.read_list(f, 'xdirection')
        ydirection = h5_util.read_list(f, 'ydirection')
        directions = [xdirection, ydirection]
    else:
        directions = [h5_util.read_list(f, 'xdirection')]

    return directions
