"""
A likelihood that has noise propotional to the observed signal (which has
to be a positive quantity).

Author:
    Ilias Bilionis

Date:
    6/10/2014

"""


__all__ = ['ProportionalNoiseLikelihood']


import numpy as np
import math
from . import Likelihood


class ProportionalNoiseLikelihood(Likelihood):

    """
    This likelihood is as follows:

    .. math::

        L(y, f(x), \\theta) = \sum_{i=1}^m \ln \mathcal{N}(y_i | f_i(x), f_i^2(x)\\theta^2).

    """

    def __init__(self, y, model, name='Proportional Likelihood'):
        """
        Initialize the object.
        """
        super(ProportionalNoiseLikelihood, self).__init__(y, model, 1,
                                                          name=name)

    def _noise_eval(self, fx, theta):
        """
        Evaluate the noise term.
        """
        m = self.model.num_output
        tmp = (self.y - fx) / (fx * theta[0])
        err2 = np.dot(tmp, tmp)
        L = (-0.5 * m * math.log(2. * math.pi)
             -m * math.log(theta[0])
             -np.sum(np.log(fx))
             -0.5 * err2)
        L_grad_theta = np.array([[-m / theta[0] + err2 / theta[0]]])
        L_grad_f = -1. / fx + tmp / (fx * theta[0]) + tmp ** 2 / fx
        L_grad_2_theta = np.array([[m / theta[0] ** 2 - 3. * err2 / theta[0] ** 2]])
        L_grad_2_f = (1. / fx ** 2
                      -1. / (fx * theta[0]) ** 2
                      -4. * tmp / (fx * theta[0]) / fx
                      -3. * (tmp / fx) ** 2)
        L_grad_2_theta_f = (-2. * tmp / (fx * theta[0]) / theta[0]
                            -2. * tmp ** 2 / (fx * theta[0]))
        state = {}
        state['L'] = L
        state['L_grad_theta'] = L_grad_theta
        state['L_grad_f'] = L_grad_f.reshape((1, m))
        state['L_grad_2_theta'] = L_grad_2_theta
        state['L_grad_2_f'] = np.diag(L_grad_2_f)
        state['L_grad_2_theta_f'] = L_grad_2_theta_f.reshape((1, m))
        return state
