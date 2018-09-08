"""
    Serialization and deserialization of directions in the direction file.
"""

import torch

def write_list(f, name, direction):
    """Write a list of numpy arrays into the hdf5 file"""
    grp = f.create_group(name)
    for i, l in enumerate(direction):
        if isinstance(l, torch.Tensor):
            l = l.numpy()
        grp.create_dataset(str(i), data=l)


def read_list(f, name):
    """Read a list of numpy arrays from the hdf5 file"""
    grp = f[name]
    return [grp[str(i)] for i in range(len(grp))]
