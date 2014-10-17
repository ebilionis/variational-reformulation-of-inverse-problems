"""
A class that represents an Gaussian likelihood with independent noise (diagonal covatiance) .

Author:
    Panagiotis Tsilifis

Date:
    5/25/2014

"""


__all__ = ['IndepNoiseGaussianLikelihood']


import numpy as np
import math
from . import Likelihood
from . import view_as_column


class IndepNoiseGaussianLikelihood(Likelihood):

    """
    The Gaussian likelihood with independent noises has as parameters the
    diagonal elements of the covariance matrix.

    It is like this:

        L(y, f(x), theta) = log N(y | f(x), diag(theta_1,...,theta_N)).

    """

    def __init__(self, y, model, name='Independent noise Gaussian Likelihood'):
        """
        Initialize the model.
        """
        num_like_params = model.num_output
        super(IndepNoiseGaussianLikelihood, self).__init__(y, model, num_like_params,
                                                          name=name)

    def _noise_eval(self, fx, theta):
        """
        Evaluate the model.
        """
        N = self.model.num_output
        tmp = np.dot(np.diag(1/theta[:]) ,self.y - fx)
        err2 = np.dot(tmp, tmp)
        L = (-0.5 * N * math.log(2. * math.pi) - np.sum(np.log(theta[:]))
              - 0.5 * err2)
        dLdfx = np.dot(np.diag(1/theta[:]), tmp)
        dLdfx = view_as_column(dLdfx)
        dLdtheta = -1 / theta[:] + tmp * tmp / theta[:]
        dLdtheta = view_as_column(dLdtheta)
        d2Ldfx2 = -np.diag(1 / theta[:] ** 2.)
        d2Ldtheta2 = 1 / theta[:] ** 2 - 3 * tmp * tmp / theta[:] ** 2
        d2Ldthetafx = -2. * tmp / theta[:] ** 2
        state = {}
        state['L'] = L
        state['L_grad_f'] = dLdfx.reshape((1, N))
        state['L_grad_theta'] = dLdtheta.reshape((1, self.num_like_params))
        state['L_grad_2_f'] = d2Ldfx2
        state['L_grad_2_theta'] = np.diag(d2Ldtheta2)
        state['L_grad_2_theta_f'] = np.diag(d2Ldthetafx)
        return state
