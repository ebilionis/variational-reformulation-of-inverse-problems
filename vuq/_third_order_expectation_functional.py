"""
A third order expectation functional.

Author:
    Ilias Bilionis

Date:
    6/5/2014

"""


__all__ = ['ThirdOrderExpectationFunctional']



import numpy as np
from . import FirstOrderExpectationFunctional
from . import MixtureOfMultivariateNormals


class ThirdOrderExpectationFunctional(FirstOrderExpectationFunctional):

    """
    A class representing a third order expectation functional over a mixture:

    .. math::

        \mathcal{F}_2[q] = \mathcal{F}_1[q] + \\frac{1}{2}\sum_{i=1}^Nw_i C_i:H_i,

    where

    .. math::

        H_i = \\nabla^2 p(\omega = \mu_i).

    """

    def __init__(self, log_p, name='Third Order Expectation Functional'):
        """
        Initialize the object.
        """
        super(ThirdOrderExpectationFunctional, self).__init__(log_p, name=name)

    def __call__(self, q):
        """
        Evaluate the functional at ``q`` and the derivatives with respect to
        all parameters.

        :returns: A dictionary containing: F1, F2, F, F_grad_w, F_grad_mu,
                  F_grad_w, F_grad_C
        """
        assert isinstance(q, MixtureOfMultivariateNormals)
        state = self._eval(q)
        F1 = state['F']
        F_grad1_w = state['F_grad_w']
        p_state = state['p_state']
        log_p_grad_2 = p_state['log_p_grad_2']
        C_H = np.einsum('ijk,ijk->i', q.C, log_p_grad_2)
        F2 = F1 + 0.5 * np.einsum('i,i', q.w, C_H)
        F_grad2_w = F_grad1_w + 0.5 * C_H[None, :] # TODO: Do we need a 0.5 on the second term?
        F_grad2_C = 0.5 * np.einsum('i,ijk->ijk', q.w, log_p_grad_2)
        state['F1'] = F1
        state['F2'] = F2
        state['F'] = F2
        state['F_grad_w'] = F_grad2_w
        state['F_grad_C'] = F_grad2_C
        return state
