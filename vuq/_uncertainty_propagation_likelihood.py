"""
A likelihood class to be used for uncertainty propagtion.

Author:
    Ilias Bilionis

Date:
    6/16/2014

"""


__all__ = ['UncertaintyPropagationLikelihood']


import numpy as np
import math
from . import Likelihood


class UncertaintyPropagationLikelihood(Likelihood):

    """
    An likelihood used for uncertainty propagation.

    It is like this:

        L(y, f(x), z, alpha) = log N(z | f(x), alpha^2 I),
    
    with y completely ignored, alpha fixed and omega = (x, z).
    
    """

    # The parameter controlling the approximation to the delta function
    _alpha = None

    @property
    def alpha(self):
        """
        :getter:    alpha
        :setter:    alpha
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        """
        Set the alpha parameter.
        """
        value = float(value)
        assert value > 0.
        self._alpha = value

    def __init__(self, model, alpha=1e-2, name='Uncertainty Propagation Likelihood'):
        """
        Initialize the object.
        """
        num_like_params = model.num_output
        self.alpha = alpha
        super(UncertaintyPropagationLikelihood, self).__init__(None, model,
                                                               num_like_params,
                                                               name=name)

    def _noise_eval(self, fx, z):
        """
        Evaluate the noise.
        """
        m = self.model.num_output
        tmp = (z - fx) / self.alpha
        err2 = np.dot(tmp, tmp)
        L = (-0.5 * m * math.log(2. * math.pi) - m * np.log(self.alpha)
             -0.5 * err2)
        L_grad_f = tmp / self.alpha
        L_grad_z = -L_grad_f
        L_grad_2_f = -np.eye(m) / self.alpha ** 2.
        L_grad_2_z = L_grad_2_f
        L_grad_2_z_f = np.eye(m) / self.alpha ** 2.
        state = {}
        state['L'] = L
        state['L_grad_f'] = L_grad_f.reshape((1, m))
        state['L_grad_theta'] = L_grad_z.reshape((1, m))
        state['L_grad_2_f'] = L_grad_2_f
        state['L_grad_2_theta'] = L_grad_2_z
        state['L_grad_2_theta_f'] = L_grad_2_z_f
        return state
