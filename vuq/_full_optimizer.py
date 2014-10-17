"""
A generic optimizer class.

Author:
    Ilias Bilionis

Date:
    6/5/2014

"""


__all__ = ['FullOptimizer']


import math
import numpy as np
from scipy.optimize import minimize
from . import EvidenceLowerBound


class FullOptimizer(object):

    """
    A generic optimizer object.

    """

    # The evidence lower bound
    _elbo = None

    # A name for the object
    __name__ = None


    @property
    def elbo(self):
        """
        :getter:    The evidence lower bound.
        """
        return self._elbo

    def __init__(self, elbo=elbo, name='Optimizer'):
        """
        Initialize the object.
        """
        assert isinstance(elbo, EvidenceLowerBound)
        self._elbo = elbo
        self.__name__ = name

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = 'Name: ' + self.__name__ + '\n'
        s += 'ELBO:\n'
        s += str(self.elbo)
        return s

    def optimize_full_mu(self, log_q):
        """
        Full optimization of the mu's.
        """
        def f_mu(mu, log_q, elbo):
            mu_old = log_q.mu[:]
            log_q.mu = mu.reshape((log_q.num_comp, log_q.num_dim))
            state = elbo(log_q)
            L = state['L1']
            L_grad_mu = state['L_grad_mu'].reshape((log_q.num_comp * log_q.num_dim, ))
            log_q.mu = mu_old
            return -L, -L_grad_mu
        args = (log_q, self.elbo)
        res = minimize(f_mu, log_q.mu.flatten(), args=args, jac=True,
                       tol=1e-20)
        log_q.mu = res.x.reshape((log_q.num_comp, log_q.num_dim))

    def optimize_full_L(self, log_q):
        """
        Full optimization of the cholesky factor of the C's.
        """
        def f_L(L, log_q, elbo, idx):
            C_old = log_q.C[:]
            k = idx[0].shape[0]
            ZZ = []
            for i in xrange(log_q.num_comp):
                Z = np.zeros((log_q.num_dim, log_q.num_dim))
                Z[idx] = L[i * k : (i + 1) * k]
                ZZ.append(Z)
                C = np.dot(Z, Z.T)
                log_q.comp[i].C = C
            ZZ = np.array(ZZ)
            state = elbo(log_q)
            LL = state['L']
            L_grad_C = state['L_grad_C']
            L_grad_Z = 2. * np.einsum('ijk,ikl->ijl', L_grad_C, ZZ)
            L_grad_Z = np.hstack([L_grad_Z[i, :, :][idx]
                                  for i in xrange(log_q.num_comp)])
            log_q.C = C_old
            print L, LL
            return -LL, -L_grad_Z
        idx = np.tril_indices(log_q.num_dim)
        L0 = np.hstack([log_q.comp[i].C[idx] for i in xrange(log_q.num_comp)])
        tmp = np.ndarray((log_q.num_dim, log_q.num_dim), dtype='object')
        for i in xrange(log_q.num_dim):
            for j in xrange(log_q.num_dim):
                if i == j:
                    tmp[i, j] = (0.5, None)
                else:
                    tmp[i, j] = (None, None)
        L_bounds = tuple(tmp[idx] for i in xrange(log_q.num_comp))
        L_bounds = tuple(xx for x in L_bounds for xx in x)
        args = (log_q, self.elbo, idx)
        res = minimize(f_L, L0, args=args, jac=True, method='L-BFGS-B',
                       bounds=L_bounds)
        k = idx[0].shape[0]
        for i in xrange(log_q.num_comp):
            Z = np.zeros((log_q.num_dim, log_q.num_dim))
            Z[idx] = res.x[i * k : (i + 1) * k]
            C = np.dot(Z, Z.T)
            log_q.comp[i].C = C

    def optimize(self, log_q, max_it=10):
        """
        Optimize.
        """
        for i in xrange(max_it):
            self.optimize_full_mu(log_q)
            self.optimize_full_L(log_q)
