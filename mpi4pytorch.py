"""                                              mpi4pytorch.py
 This module contains convenience methods that make it easy to use mpi4py.  The available functions handle memory
 allocation and other data formatting tasks so that tensors can be easily reduced/broadcast using 1 line of code.
"""

import numpy as np
import mpi4py

def setup_MPI():
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        #  Convert the Object to a Class so that it is possible to add attributes later
        class A(mpi4py.MPI.Intracomm):
            pass
        comm = A(comm)
    except:
       comm = None

    return comm


def print_once(comm, *message):
    if not comm or comm.Get_rank()==0:
        print (''.join(str(i) for i in message))

def is_master(comm):
    return not comm or comm.Get_rank()==0

def allreduce_max(comm, array, display_info=False):
    if not comm:
        return array
    array = np.asarray(array, dtype='d')
    total = np.zeros_like(array)
    float_min = np.finfo(np.float).min
    total.fill(float_min)

    if display_info:
        print ("(%d): sum=%f : size=%d"%(get_rank(comm), np.sum(array), array.nbytes))
        rows = str(comm.gather(array.shape[0]))
        cols = str(comm.gather(array.shape[1]))
        print_once(comm, "reduce: %s, %s"%(rows, cols))

    comm.Allreduce(array, total, op=mpi4py.MPI.MAX)
    return total

def allreduce_min(comm, array, display_info=False):
    if not comm:
        return array
    array = np.asarray(array, dtype='d')
    total = np.zeros_like(array)
    float_max = np.finfo(np.float).max
    total.fill(float_max)

    if display_info:
        print ("(%d): sum=%f : size=%d"%(get_rank(comm), np.sum(array), array.nbytes))
        rows = str(comm.gather(array.shape[0]))
        cols = str(comm.gather(array.shape[1]))
        print_once(comm, "reduce: %s, %s"%(rows, cols))

    comm.Allreduce(array, total, op=mpi4py.MPI.MIN)
    return total


def reduce_max(comm, array, display_info=False):
    if not comm:
        return array
    array = np.asarray(array, dtype='d')
    total = np.zeros_like(array)
    float_min = np.finfo(np.float).min
    total.fill(float_min)

    if display_info:
        print ("(%d): sum=%f : size=%d"%(get_rank(comm), np.sum(array), array.nbytes))
        rows = str(comm.gather(array.shape[0]))
        cols = str(comm.gather(array.shape[1]))
        print_once(comm, "reduce: %s, %s"%(rows, cols))

    comm.Reduce(array, total, op=mpi4py.MPI.MAX, root=0)
    return total

def reduce_min(comm, array, display_info=False):
    if not comm:
        return array
    array = np.asarray(array, dtype='d')
    total = np.zeros_like(array)
    float_max = np.finfo(np.float).max
    total.fill(float_max)

    if display_info:
        print ("(%d): sum=%f : size=%d"%(get_rank(comm), np.sum(array), array.nbytes))
        rows = str(comm.gather(array.shape[0]))
        cols = str(comm.gather(array.shape[1]))
        print_once(comm, "reduce: %s, %s"%(rows, cols))

    comm.Reduce(array, total, op=mpi4py.MPI.MIN, root=0)
    return total

def barrier(comm):
    if not comm:
        return
    comm.barrier()

def get_mpi_info():
    try:
        return mpi4py.MPI.get_vendor()
    except ImportError:
        return "none"

def get_rank(comm):
    try:
        return comm.Get_rank()
    except ImportError:
        return 0

def get_num_procs(comm):
    try:
        return comm.Get_size()
    except ImportError:
        return 1
