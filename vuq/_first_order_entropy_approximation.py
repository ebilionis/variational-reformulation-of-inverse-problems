"""
A first order approximation to the entropy.

Author:
    Ilias Bilionis

Date:
    6/3/2014

"""


__all__ = ['FirstOrderEntropyApproximation']


import numpy as np
from . import MixtureOfMultivariateNormals
from . import EntropyApproximation


class FirstOrderEntropyApproximation(EntropyApproximation):

    """
    See :class:`vuq.EntropyApproximation` for the documentation.

    The first order entropy approximation is:

    .. math::
        S_1[q] = - \sum_{i=1}^N w_i \ln(q(\mu_i)).

    """

    def __init__(self, name='First Order Entropy Approximation'):
        """
        Initialize the object.
        """
        super(FirstOrderEntropyApproximation, self).__init__(name=name)

    def eval(self, q):
        """
        Evaluate only the entropy (no derivatives).
        """
        assert isinstance(q, MixtureOfMultivariateNormals)
        w = q.w
        mu = q.mu
        log_q = q(mu)
        S0 = -np.sum(w * log_q) + 0.5 * q.num_dim
        return S0

    def __call__(self, q):
        """
        Evaluate the entropy and the derivatives.
        """
        # Sanity check
        assert isinstance(q, MixtureOfMultivariateNormals)
        w = q.w
        mu = q.mu
        log_q, grad_x_log_q, grad_mu_log_q, grad_C_log_q = q.log_q_grad_x_mu_and_C(mu)
        # The approximation (TODO: is it ok to add the last term?)
        S0 = -np.sum(w * log_q)
        # The gradient with respect to w
        grad_w = -log_q
        # The gradient with respect to mu
        grad_mu = -np.einsum('i,ijk->ijk', w, grad_x_log_q)
        # The gradient with respect to C
        grad_mu -= np.einsum('i,ijkl->jkl', w, grad_mu_log_q)
        grad_C = -np.einsum('i,ijkl->jkl', w, grad_C_log_q)
        state = {}
        state['S'] = S0 
        state['S_grad_w'] = grad_w
        state['S_grad_mu'] = grad_mu
        state['S_grad_C'] = grad_C
        return state
