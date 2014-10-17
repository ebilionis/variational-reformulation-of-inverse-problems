"""
Some routines that are common to all scripts.

Author:
    Ilias Bilionis

Date:
    9/9/2014

"""


import sys
import os
import imp
import cPickle as pickle
import numpy as np
from optparse import OptionParser


# Make sure that we can see the path that shows to the base of this project
project_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..'))
# And add this to our path
sys.path.insert(0, project_dir)

# Now try to load vuq
import vuq


# Now some useful function definitions

def init_mpi(use_mpi):
    """
    Initialize mpi (if requested).
    """
    if use_mpi is not None:
        import mpi4py.MPI as mpi
        comm = mpi.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        return mpi, comm, rank, size
    else:
        return None, None, 0, 1


def get_rank_size(comm):
    """
    Get the rank and size from a communicator.
    """
    if comm is None:
        return 0, 1
    rank = comm.Get_rank()
    size = comm.Get_size()
    return rank, size


def print_once(msg, comm=None):
    """
    Use to print a message once when using mpi.
    """
    if comm is None:
        sys.stdout.write(msg)
    else:
        rank = comm.Get_rank()
        if rank == 0:
            sys.stdout.write(msg)
            sys.stdout.flush()
        comm.barrier()


def initialize_model(model_file, model_name='Model', comm=None):
    """
    Initialize the ``model_file`` which should be a proper python module.
    """
    rank, size = get_rank_size(comm)
    print_once('Initializing the %s.\n' % model_name, comm=comm)
    print_once('-------------------------\n', comm=comm)
    for i in xrange(size):
        if rank == i:
            try:
                model = imp.load_source('', model_file)
            except Exception as e:
                signal_fatal_error('I couldn\'t load the %s.\n' % model_name,
                                   comm=comm, e=e)
            print_rank(i, 'initialized %s.\n' % model_name)
        mpi_wait(comm)
    print_once('Done.\n'
               '-------------------------\n', comm=comm)
    return model


def initialize_pymc_model(model_file, model_name='PyMC Model', comm=None):
    """
    Initialize a PyMC model.
    """
    return initialize_model(model_file, model_name=model_name, comm=comm).make_model()


def initialize_native_model(model_file, model_name='Native Model', comm=None):
    """
    Initialize a native model.
    """
    return initialize_model(model_file, model_name=model_name, comm=comm).make_model()


def signal_fatal_error(err, err_code=1, comm=None, e=None):
    """
    Signal a fatal error. The program will exit.
    """
    sys.stderr.write(err)
    if comm is None:
        if e is not None:
            print e
        sys.exit(1)
    else:
        if comm.Get_rank() == 0:
            print e
        comm.Abort(1)


def print_rank(i, msg):
    """
    Use to print information from a particular rank.
    This is an mpi only function.
    """
    sys.stdout.write('rank %d: %s' % (i, msg))
    sys.stdout.flush()


def mpi_wait(comm):
    """
    Wait if comm is not ``None``.
    """
    if comm is not None:
        comm.barrier()


def get_running_statistics(data):
    """
    Returns the running mean and standard deviation of the data.
    """
    x_m = np.cumsum(data, axis=0) / np.arange(1, data.shape[0] + 1)[:, None]
    x2_m = np.cumsum(data ** 2., axis=0) / np.arange(1, data.shape[0] + 1)[:, None]
    x_s = np.sqrt(x2_m - x_m ** 2.)
    return x_m, x_s
