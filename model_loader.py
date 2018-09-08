import os
import cifar10.model_loader

def load(dataset, model_name, model_file, data_parallel=False):
    assert os.path.exists(model_file), model_file + " does not exist."
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    return net
