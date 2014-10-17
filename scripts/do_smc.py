"""
Solves the inverse problem using Sequential Monte Carlo.

Author:
    Ilias Bilionis

Date:
    9/9/2014

"""


from common import *
import pymc as pm
import pysmc as ps


def main(options):
    """
    The main function.
    """
    mpi, comm, rank, size = init_mpi(options.use_mpi)
    model = initialize_pymc_model(options.model, comm=comm)
    mcmc = pm.MCMC(model)
    for rv in mcmc.stochastics:
        mcmc.use_step_method(ps.RandomWalk, rv)
    if options.num_particles is None:
        options.num_particles = size
    if options.db_filename is None:
        db_filename = (os.path.splitext(options.model)[0] +
                       '_N=' + str(options.num_particles) +
                       '_M=' + str(options.num_mcmc) + '.pcl')
    if rank == 0 and os.path.exists(db_filename):
        print '-', db_filename, 'exists'
        print '- I am removing it'
        os.remove(db_filename)
    smc_sampler = ps.SMC(mcmc,
                         num_particles=options.num_particles,
                         num_mcmc=options.num_mcmc,
                         verbose=4,
                         db_filename=db_filename,
                         gamma_is_an_exponent=True,
                         mpi=mpi,
                         update_db=True)
    smc_sampler.initialize(0.)
    smc_sampler.move_to(1.)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-m', '--model', dest='model', metavar='FILE',
                      help='set the file containing the model to sample from (PyMC model)')
    parser.add_option('--use-mpi', dest='use_mpi', action='store_true',
                      help='use mpi or not') 
    parser.add_option('--num-particles', dest='num_particles', type=float,
                      help='the number of particles you want to use')
    parser.add_option('--num-mcmc', dest='num_mcmc', type=float, default=1,
                      help='the number of MCMC steps per gamma')
    parser.add_option('--db-filename', dest='db_filename',
                      help='the name of the database')
    options, args = parser.parse_args()
    main(options)
