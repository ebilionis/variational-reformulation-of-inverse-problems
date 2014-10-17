"""
A lower bound to the entropy derived by making use of Jensen's inequality.

Author:
    Ilias Bilionis

Date:
    16/9/2014

"""


__all__ = ['EntropyLowerBound']


import numpy as np
import scipy
import math
from scipy.misc import logsumexp
from . import MixtureOfMultivariateNormals
from . import EntropyApproximation


class EntropyLowerBound(EntropyApproximation):

    """
    See :class:`vuq.EntropyApproximation` for the documentation.

    The lower bound to the entropy is:

    .. math::
        S_l[q] = \sum_{i=1}^N w_i q_i,

    where

    .. math::
        q_i = \sum_{j=1}^N w_j N_{ij}.

    """

    def __init__(self, name='Entropy Lower Bound'):
        """
        Initialize the object.
        """
        super(EntropyLowerBound, self).__init__(name=name)

    def eval(self, q):
        """
        Evaluate only the entropy (no derivatives).
        """
        assert isinstance(q, MixtureOfMultivariateNormals)
        num_dim = q.num_dim
        log_w = q.log_w
        w = q.w
        mu = q.mu
        C = q.C
        N = q.num_comp
        # Compute the Cholesky decomposition of all pairs.
        L_conv = []
        inv_C_conv = []
        log_det_C_conv = []
        log_convs = np.zeros((N, N))
        for i in xrange(N):
            for j in xrange(i, N):
                L = scipy.linalg.cho_factor(C[i, :, :] + C[j, :, :], lower=True)
                L_conv.append(L)
                inv_C_conv.append(scipy.linalg.cho_solve(L, np.eye(num_dim)))
                log_det_C_conv.append(2. * np.sum(np.log(np.diag(L[0]))))
                t = scipy.linalg.solve_triangular(L[0], mu[i, :] - mu[j, :],
                								  lower=L[1])
                log_convs[i, j] = (0.5 * num_dim * math.log(2. * math.pi)
                               - 0.5 * log_det_C_conv[-1] - 0.5 * np.dot(t.T, t))
                log_convs[j, i] = log_convs[i, j]
        log_qus = logsumexp(log_convs + log_w, axis=1)
        Sl = -np.sum(w * log_qus)
        S_grad_w = -log_qus - np.exp(logsumexp(log_convs + log_w - log_qus, axis=1))
        e = np.zeros((N, N, num_dim))
        count = 0
        for i in xrange(N):
        	for j in xrange(i, N):
        		e[i, j, :] = scipy.linalg.cho_solve(L_conv[count],
        									        mu[i, :] - mu[j, :])
        		e[j, i, :] = -e[i, j, :]
        		count += 1
        A = np.zeros((N, N, num_dim, num_dim))
        count = 0
        for i in xrange(N):
        	for j in xrange(i, N):
        		for m in xrange(num_dim):
        			for n in xrange(num_dim):
        				A[i, j, m, n] = (inv_C_conv[count][m, n] -
        								 e[i, j, m] * e[i, j, n])
        				A[j, i, m, n] = A[i, j, m, n]
        		count += 1
    	S_grad_mu = np.zeros((N, 1, num_dim))
    	for k in xrange(N):
    		for m in xrange(num_dim):
    			S_grad_mu[k, 0, m] = 0.
    			for i in xrange(N):
    				S_grad_mu[k, 0, m] += (w[i] *
    								       math.exp(log_convs[i, k]) *
    									   e[i, k, m] *
    									   (1. / math.exp(log_qus[k]) +
    									    1. / math.exp(log_qus[i])))
    			S_grad_mu[k, 0, m] *= - w[k]
    	S_grad_C = np.zeros((N, num_dim, num_dim))
    	for k in xrange(N):
    		for m in xrange(num_dim):
    			for n in xrange(m, num_dim):
    				S_grad_C[k, m, n] = 0.
    				for i in xrange(N):
    					S_grad_C[k, m, n] += (w[i] *
    										  math.exp(log_convs[i, k]) *
    										  A[k, i, m, n] *
    										  (1. / math.exp(log_qus[k]) +
    									       1. / math.exp(log_qus[i])))
    				S_grad_C[k, m, n] *= 0.5 * w[k]
    			S_grad_C[k, n, m] = S_grad_C[k, m, n]
        return Sl, S_grad_mu, S_grad_C, S_grad_w

    def __call__(self, q):
        """
        Evaluate the entropy and the derivatives.
        """
        S0, S_grad_mu, S_grad_C, S_grad_w = self.eval(q)
        state = {}
        state['S'] = S0
        state['S_grad_w'] = S_grad_w
        state['S_grad_mu'] = S_grad_mu
        state['S_grad_C'] = S_grad_C
        return state