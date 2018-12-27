"""                                          mpi4pytorch_placeholder.py
 This module has the same interface as mpi4pytorch, however the methods "do nothing," i.e., they make no calls to MPI
 routines and they return results as if there is only 1 single process running.  This module can be imported
 instead of 'mpi4pytorch' on systems where 'mpi4py' is not installed. This module serves as a place-holder, enabling
 reduce/broadcast methods to be called despite the fact that there is only 1 process running.
"""

import numpy as np
from mpi4py import MPI

def setup_MPI():
    return None

def print_once(comm, *message):
    print (''.join(str(i) for i in message))

def is_master(comm):
    return True

def allreduce_max(comm, array, display_info=False):
    return array

def allreduce_min(comm, array, display_info=False):
    return array

def reduce_max(comm, array, display_info=False):
    return array

def reduce_min(comm, array, display_info=False):
    return array

def barrier(comm):
    return

def get_mpi_info():
    return "none"

def get_rank(comm):
    return 0

def get_num_procs(comm):
    return 1
