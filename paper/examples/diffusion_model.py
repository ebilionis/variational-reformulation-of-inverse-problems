"""
The classic diffusion model.

Author:
    Ilias Bilionis

Date:
    9/13/2014

"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.split(__file__)[0]))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0], '../..')))
from diffusion_common import *
from diffusion import ContaminantTransportModel
from vuq import *


def make_model():
    """
    Make the diffusion model.
    """
    # The number of dimensions for this particular problem
    num_dim = 3

    # The log prior
    log_prior = PDFCollection([UniformND(2),
                               MultivariateNormal([[-1.]])])

    # The data
    y = load_diffusion_data()

    # The forward model
    solver = ContaminantTransportModel()

    # The log likelihood
    log_like = IsotropicGaussianLikelihood(y, solver)

    # The joint
    log_p = Joint(log_like, log_prior)

    return locals()


if __name__ == '__main__':
    model = make_model()
    print model
