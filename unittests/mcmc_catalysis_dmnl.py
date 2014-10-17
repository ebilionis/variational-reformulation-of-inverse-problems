"""
Solve the catalysis problem using mcmc.

Author:
    Ilias Bilionis

Date:
    6/26/2014

"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import mpi4py.MPI as mpi
import pymc as pm
import pysmc as ps
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'.')
sys.path.insert(0,'demos/')
from catalysis import CatalysisModelDMNLESS



def make_model():
    gamma = 1.
    kappa = pm.Gamma('kappa', 4., 1., size=5)
    sigma2 = pm.Gamma('sigma2', 0.1, 1., value=100.)
    data = np.loadtxt('data.txt').reshape((7, 6))
    y = data[:, 1:]
    y = y.reshape((1, y.shape[0] * y.shape[1])) / 500.
    f = CatalysisModelDMNLESS()
    @pm.deterministic
    def model_output(kappa=kappa):
        return f(kappa)['f']
    @pm.stochastic(observed=True)
    def output(value=y, model_output=model_output, sigma2=sigma2, gamma=gamma):
        return gamma * pm.normal_like(y, model_output, 1. / sigma2)
    return locals()


if __name__ == '__main__':
    model = make_model()
    mcmc = pm.MCMC(model)
    mcmc.use_step_method(ps.RandomWalk, model['kappa'])
    mcmc.use_step_method(ps.RandomWalk, model['sigma2'])
    smc_sampler = ps.SMC(mcmc, num_particles=10240, num_mcmc=1, verbose=4,
                         db_filename='demos/smc_catalysis.pcl',
                         gamma_is_an_exponent=True, mpi=mpi,
                         update_db=True)
    smc_sampler.initialize(0.)
    smc_sampler.move_to(1.)
