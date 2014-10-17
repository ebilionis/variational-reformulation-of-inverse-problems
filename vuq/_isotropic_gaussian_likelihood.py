"""
A class that represents an isotropic Gaussian likelihood.

Author:
    Ilias Bilionis

Date:
    5/23/2014

"""


__all__ = ['IsotropicGaussianLikelihood']


import numpy as np
import math
from scipy.sparse import diags
from . import Likelihood
from . import view_as_column


class IsotropicGaussianLikelihood(Likelihood):

    """
    The isotropic Gaussian likelihood has only one parameter.

    It is like this:

        L(y, f(x), theta) = log N(y | f(x), theta I).

    """

    def __init__(self, y, model, name='Isotropic Gaussian Likelihood'):
        """
        Initialize the model.
        """
        num_like_params = 1
        super(IsotropicGaussianLikelihood, self).__init__(y, model, num_like_params,
                                                          name=name)

    def _noise_eval(self, fx, theta):
        """
        Evaluate the model.
        """
        N = self.model.num_output
        sigma = math.exp(theta[0])
        tmp = (self.y - fx) / sigma
        err2 = np.dot(tmp, tmp)
        L = (-0.5 * N * math.log(2. * math.pi) - N * np.log(sigma)
             -0.5 * err2)
        L_grad_f = tmp / sigma
        L_grad_f = L_grad_f.reshape((1, N))
        L_grad_theta = np.array([[-N / sigma + err2 / sigma]])
        #L_grad_2_f = diags(-np.ones(N) / sigma ** 2, 0, format='lil')
        L_grad_2_f = -np.eye(N) / sigma ** 2.
        L_grad_2_theta = np.array([[N / sigma - 3. * err2 / sigma ** 2]])
        L_grad_2_theta_f = -2. * tmp / sigma ** 2
        L_grad_2_theta_f = L_grad_2_theta_f.reshape((1, N))
        state = {}
        state['L'] = L
        state['L_grad_f'] = L_grad_f
        state['L_grad_theta'] = L_grad_theta * sigma
        state['L_grad_2_f'] = L_grad_2_f
        state['L_grad_2_theta'] = L_grad_2_theta * (sigma ** 2.)
        state['L_grad_2_theta_f'] = L_grad_2_theta_f
        return state
