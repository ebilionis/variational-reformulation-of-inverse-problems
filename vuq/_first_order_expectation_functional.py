"""
A first order expectation functional.

Author:
    Ilias Bilionis

Date:
    6/5/2014

"""


__all__ = ['FirstOrderExpectationFunctional']


import numpy as np
from . import ExpectationFunctional
from . import MixtureOfMultivariateNormals


class FirstOrderExpectationFunctional(ExpectationFunctional):

    """
    A class representing a first order expectation functional over a mixture:

    .. math::

        \mathcal{F}_1[q] = \sum_{i=1}^N w_i \ln p(\omega = \mu_i).

    """

    def __init__(self, log_p, name='First Order Expectation Functional'):
        """
        Initialize the object.
        """
        super(FirstOrderExpectationFunctional, self).__init__(log_p, name=name)

    def _eval(self, q):
        """
        Evaluate the functional and its derivatives at ``q`` and propagate
        forward any data coming from log_p.
        """
        p_state = self.log_p.eval_all(q.mu)
        log_p = p_state['log_p']
        log_p_grad = p_state['log_p_grad']
        F1 = np.sum(q.w * log_p)
        F_grad1_w = log_p
        F_grad1_mu = np.einsum('i,ijk->ijk', q.w, log_p_grad)
        state = {}
        state['F'] = F1
        state['F_grad_w'] = F_grad1_w
        state['F_grad_mu'] = F_grad1_mu
        state['F_grad_C'] = np.zeros(q.C.shape)
        state['p_state'] = p_state
        return state

    def __call__(self, q):
        """
        Evaluate the functional at ``q`` and the derivatives with respect to
        all parameters.

        :returns: A dictionary containing the following: F, F_grad_w,
                  F_grad_mu, tmp.
        """
        # Sanity check
        assert isinstance(q, MixtureOfMultivariateNormals)
        assert q.num_dim == self.log_p.num_dim
        return self._eval(q)
