"""
A generic optimizer class.

Author:
    Ilias Bilionis

Date:
    6/5/2014

"""


__all__ = ['Optimizer']


import math
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from . import EvidenceLowerBound


class Optimizer(object):

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

    def optimize_mu(self, log_q, i, mu_bounds, mu_constraints):
        """
        Optimize one mu.
        """
        def f_mu(mu, log_q, elbo, i):
            print 'mu', mu
            old_mu_i = np.copy(log_q.comp[i].mu)
            log_q.comp[i].mu = mu[:, None]
            state = elbo(log_q)
            log_q.comp[i].mu = old_mu_i
            L = state['L1']
            L_grad_mu = state['L_grad_mu'][i, 0, :]
            return -L, -L_grad_mu
        args = (log_q, self.elbo, i)
        res = minimize(f_mu, log_q.comp[i].mu.flatten(), args=args,
                       jac=True, method='L-BFGS-B', bounds=mu_bounds)
        print res
        log_q.comp[i].mu = res.x[:, None]
        return -res.fun, res.nfev

    def optimize_full_mu(self, log_q, mu_bounds):
        """
        Full optimization of the mu's.
        """
        if mu_bounds is not None:
            if not len(mu_bounds) == log_q.num_comp * log_q.num_dim:
                tmp = tuple(mu_bounds for i in xrange(log_q.num_comp))
                mu_bounds = tuple(x for xx in tmp for x in xx)
        def f_mu(mu, log_q, elbo):
            print 'mu:', mu
            mu_old = log_q.mu[:]
            log_q.mu = mu.reshape((log_q.num_comp, log_q.num_dim))
            state = elbo(log_q)
            L = state['L1']
            L_grad_mu = state['L_grad_mu'].reshape((log_q.num_comp * log_q.num_dim, ))
            log_q.mu = mu_old
            return -L, -L_grad_mu
        args = (log_q, self.elbo)
        res = minimize(f_mu, log_q.mu.flatten(), args=args, jac=True,
                       bounds=mu_bounds, method='L-BFGS-B')
        print res
        log_q.mu = res.x.reshape((log_q.num_comp, log_q.num_dim))
        return -res.fun, res.nfev

    def optimize_C(self, log_q):
        """
        Optmize all the C's together.
        """
        def f_c_diag(c, log_q, elbo):
            print 'C', c
            old_C = np.copy(log_q.C)
            for i in xrange(log_q.num_comp):
                log_q.comp[i].C = np.eye(log_q.num_dim) * c[i]
            state = elbo(log_q)
            for i in xrange(log_q.num_comp):
                log_q.comp[i].C = np.eye(log_q.num_dim) * old_C[i]
            L = state['L']
            L_grad_c = np.einsum('ijj->i', state['L_grad_C'])
            print L
            return -L, -L_grad_c
        args = (log_q, self.elbo)
        bounds = tuple((1e-6, 10.) for i in xrange(log_q.num_comp))
        res = minimize(f_c_diag, log_q.C[:, 0, 0], args=args,
                       jac=True, bounds=bounds,
                       method='L-BFGS-B')
        for i in xrange(log_q.num_comp):
            log_q.comp[i].C = np.eye(log_q.num_dim) * res.x[i]
        return -res.fun

    def optimize_C_diag(self, log_q, C_bounds):
        """
        Optimize all the C's together as if you had diagonal covariance matrices.
        """
        if C_bounds is None:
            C_bounds = tuple((1e-3, 1e3)
                             for i in xrange(log_q.num_comp * log_q.num_dim))
        def f_c_diag(c, log_q, elbo):
            print 'c:', c
            num_dim = log_q.num_dim
            old_C = np.copy(log_q.C)
            for i in xrange(log_q.num_comp):
                log_q.comp[i].C = np.diag(c[i * num_dim:(i + 1) * num_dim])
            state = elbo(log_q)
            log_q.C = old_C
            L = state['L']
            L_grad_c = np.einsum('ijj->ij', state['L_grad_C']).flatten()
            return -L, -L_grad_c
        args = (log_q, self.elbo)
        c_init = np.einsum('ijj->ij', log_q.C).flatten()
        if not len(C_bounds) == c_init.shape[0]:
            C_bounds = C_bounds * log_q.num_comp
            assert len(C_bounds) == c_init.shape[0]
        res = minimize(f_c_diag, np.einsum('ijj->ij', log_q.C).flatten(), args=args,
                       jac=True, bounds=C_bounds,
                       method='L-BFGS-B')
        print res
        num_dim = log_q.num_dim
        for i in xrange(log_q.num_comp):
            log_q.comp[i].C = np.diag(res.x[i * num_dim:(i + 1) * num_dim])
        return -res.fun

    def optimize_w(self, log_q):
        def f_w(w, log_q, elbo):
            print 'w', w, np.sum(w)
            w_old = log_q.w[:]
            log_q.w = w
            state = elbo(log_q)
            L = state['L']
            L_grad_w = state['L_grad_w'].flatten()
            log_q.w = w_old
            return -L, -L_grad_w
        args = (log_q, self.elbo)
        def w_c_fun(w):
            return np.sum(w) - 1.
        def w_c_jac(w):
            return np.ones(w.shape)
        w_c = {}
        w_c['type'] = 'eq'
        w_c['fun'] = w_c_fun
        w_c['jac'] = w_c_jac
        w_c['args'] = ()
        w_bounds = tuple((1e-6, 1.-1e-6) for i in xrange(log_q.num_comp))
        res = minimize(f_w, log_q.w, args=args, jac=True,
                       method='SLSQP', bounds=w_bounds, constraints=w_c)
        print res
        log_q.w = res.x
        return -res.fun

    def optimize(self, log_q, tol=1e-3, max_it=1000,
                 mu_bounds=None, mu_constraints=None, C_bounds=None,
                 full_mu=False):
        """
        Optimize starting from log_q.

        All changes are done in place.
        """
        # Get the elbo here
        state = self.elbo(log_q)
        L_current = state['L']
        # Loop
        L = []
        nfev = 0
        for it in xrange(max_it):
            prev_log_q = deepcopy(log_q)
            if full_mu:
                LL, nfevc = self.optimize_full_mu(log_q, mu_bounds)
                nfev += log_q.num_comp * nfevc
            else:
                # Optimize the mu's sequentially
                for i in xrange(log_q.num_comp):
                    LL, nfevc = self.optimize_mu(log_q, i, mu_bounds=mu_bounds,
                                                 mu_constraints=None)
                    nfev += nfevc
            #L.append(self.optimize_w(log_q))
            # Optimize the C's all together
            L.append(self.optimize_C_diag(log_q, C_bounds=C_bounds))
            # Check convergence
            L_old = L_current
            L_current = L[-1]
            print it, L_current, L_current - L_old
            if math.fabs(L_current - L_old) < tol:
                print 'Converged ***'
                break
        return L, nfev
