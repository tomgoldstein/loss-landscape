"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd.variable import Variable

def eval_loss(net, criterion, loader, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0 # number of samples
    num_batch = len(loader)

    if use_cuda:
        net.cuda()
    net.eval()

    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)
                targets = Variable(targets)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()*batch_size
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets.data).sum()

        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)

                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)
                if use_cuda:
                    inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum()

    return total_loss/total, 100.*correct/total


def eval_grad(net, criterion, loader, rank=0, use_cuda=False):
    """
    Evaluate the gradient for training model 'net' with the data defined
    by the loader.

    Args:

    Returns:
      averaged gradient
    """

    if use_cuda:
        net.cuda()
    net.eval()
    net.zero_grad()

    num_batch = len(loader)
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

    return net


def eval_hessian_product(vec, params, net, criterion, loader, verbose=False, rank=0,
                         use_cuda=False, is_float=False):
    """
    Evaluate product of the Hessian of the loss function with a direction vector "vec".
    The input "vec" should be a list of Variable's with the same dimensions as the parameters.
    """

    if use_cuda:
        net.cuda()
        if is_float:
            vec = [v.cuda().float() for v in vec]
        else:
            vec = [v.cuda() for v in vec]

    net.eval()
    net.zero_grad()

    num_batch = len(loader)
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        start_time = time.time()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        forward_time = time.time() - start_time

        start_time = time.time()
        grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True)
        grad_time = time.time() - start_time

        # Compute inner product of gradient with the direction vector
        prod = Variable(torch.zeros(1)).type(type(grad_f[0].data))
        for (g, v) in zip(grad_f, vec):
            prod = prod + (g * v).cpu().sum()

        # Compute the Hessian-vector product, H*v
        # backward() will accumulate the gradients into the .grad attributes
        start_time = time.time()
        prod.backward()
        backward_time = time.time() - start_time

        # print "forward: %f grad: %f backward: %f" % (forward_time, grad_time, backward_time)

    return net
