"""
A dimensionless version of the catalysis model expressed in PyMC.

Author:
    Ilias Bilionis

Date:
    9/9/2014

"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.split(__file__)[0]))
from catalysis_common import *
import pymc as pm
from catalysis import CatalysisModelDMNLESS


def make_model(kappa_a=1., kappa_b=1., sigma_a=1., sigma_b=.1):
    """
    A PyMC version of the dimensionless catalysis model.
    """
    gamma = 1.
    log_kappa = pm.Normal('log_kappa', 0., 1., size=5)
    log_sigma = pm.Normal('log_sigma', -1., 1.)
    kappa = pm.exp(log_kappa)
    sigma = pm.exp(log_sigma)
    y = load_dimensionless_catalysis_data()
    f = CatalysisModelDMNLESS()
    @pm.deterministic
    def model_output(log_kappa=log_kappa):
        return f(log_kappa)['f']
    @pm.stochastic(observed=True)
    def output(value=y, model_output=model_output, sigma=sigma, gamma=gamma):
        return gamma * pm.normal_like(y, model_output, 1. / (sigma ** 2.))
    return locals()


if __name__ == '__main__':
    """
    A simple test for the model.
    """
    import pysmc as ps
    model = make_model()
    mcmc = pm.MCMC(model)
    for rv in mcmc.stochastics:
        mcmc.use_step_method(ps.RandomWalk, rv)
    print model
