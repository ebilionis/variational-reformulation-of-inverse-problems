"""
The native model for the dimensionless catalysis problem.

Author:
    Ilias Bilionis

Date:
    9/9/2014

"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.split(__file__)[0]))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0], '../..')))
from catalysis_common import *
from vuq import MultivariateNormal
from vuq import PDFCollection
from vuq import Joint
from vuq import IsotropicGaussianLikelihood
from catalysis import CatalysisModelDMNLESS


def make_model():
    """
    The native version of the dimensionless catalysis model.
    """
    # The number of dimensions for this particular problem
    num_dim = 6

    # The log prior
    prior_mu = np.hstack([np.zeros(num_dim - 1), -1.])
    prior_C = np.eye(num_dim)
    log_prior = MultivariateNormal(prior_mu, C=prior_C)

    # The data
    y = load_dimensionless_catalysis_data()

    # The forward model
    catal_model = CatalysisModelDMNLESS()

    # The log likelihood
    log_like = IsotropicGaussianLikelihood(y[0, :], catal_model)

    # The joint
    log_p = Joint(log_like, log_prior)

    return locals()


if __name__ == '__main__':
    model = make_model()
    print model
