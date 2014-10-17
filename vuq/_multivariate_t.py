"""
A multivariate Student-t PDF.

Author:
    Panagiotis Tsilifis

Date:
    6/5/2014

"""


__all__ = ['MultivariateT']


import numpy as np
import math
import scipy.linalg
from scipy import special
from . import make_vector
from . import call_many
from . import PDFBase


class MultivariateT(PDFBase):

    """
    A class representing the PDF of a multivariate Normal distribution.

    :param mu:  The location of the distribution.
    :type mu:   :class:`numpy.ndarray`
    :param C:   The scale matrix. It is taken to be the unit matrix,
                if it is not specified.
    :type C:    :class:`numpy.ndarray`

    :param nu:  The degrees of freedom
    :type nu:   Integer

    """

    # The location
    _mu = None

    # The scale matrix
    _C = None

    # The degrees of freedom
    _nu = None

    # The Cholesky decomposition of C
    _L = None

    # The log of determinant of C
    _log_det_C = None

    # The inverse of C
    _inv_C = None

    @property
    def mu(self):
        """
        :getter:    The location of the distribution. Internally, it is represented
                    as a row matrix.
        :setter:    Set mu.
        """
        return self._mu

    @mu.setter
    def mu(self, value):
        """
        Set mu.
        """
        value = make_vector(value)
        assert value.shape[0] == self.num_dim
        self._mu = value

    @property
    def C(self):
        """
        :getter:    The scale matrix.
        :setter:    Set scale matrix.
        """
        return self._C

    @C.setter
    def C(self, value):
        """
        Set the covariance matrix
        """
        assert value.ndim == 2
        assert value.shape[0] == self.num_dim and value.shape[1] == self.num_dim
        self._C = value
        # If the following fails, then we have a rank defficient covariance
        try:
            self._L = scipy.linalg.cho_factor(self.C, lower=True)
            self._inv_C = scipy.linalg.cho_solve(self.L, np.eye(self.num_dim))
            self._log_det_C = 2. * np.sum(np.log(np.diag(self.L[0])))
        except scipy.linalg.LinAlgError as e:
            # In this case, we need to find any matrix L such that C = L * L^T.
            # Only sampling will work. The log PDF, the gradient and the Hessian are
            # garbage in this case.
            self._inv_C = np.zeros((self.num_dim, self.num_dim))
            self._log_det_C = 0.
            lam, V = scipy.linalg.eigh(self.C)
            idx = lam > 1e-10
            lam = lam[idx]
            V = V[:, idx]
            L = np.dot(V, np.diag(np.sqrt(lam)))
            Cp = np.dot(L, L.T)
            self._L = (L, None)

    @property
    def L(self):
        """
        :getter:    The Cholesky decomposition of C.
        """
        return self._L

    @property
    def log_det_C(self):
        """
        :getter:    The logarithm of the determinant of ``C``.
        """
        return self._log_det_C

    @property
    def inv_C(self):
        """
        :getter:    The inverse of ``C``.
        """
        return self._inv_C

    @property
    def nu(self):
        """
        :getter:    The degrees of freedom.
        :setter:    Set nu.
        """
        return self._nu

    @nu.setter
    def nu(self, value):
        """
        Set nu.
        """
        #assert isinstance(value, int)
        self._nu = value

    def __init__(self, mu, nu, C=None, name='Multivariate Student-t'):
        """
        Initialize the object.
        """
        self._mu = make_vector(mu)
        super(MultivariateT, self).__init__(self.mu.shape[0], name=name)
        if C is None:
            C = np.eye(self.num_dim)
        self.C = C
        self.nu = nu
        # const is the part of the likelihood that does not depend on any parameter
        self._const = -0.5 * self.num_dim * math.log(math.pi)

    def _eval(self, x):
        """
        Evaluate the log of the PDF at x.
        """
        t = scipy.linalg.solve_triangular(self.L[0], self.mu - x, lower=self.L[1])
        z1 = -np.log(special.gamma(self.nu / 2.))
        z2 = -0.5 * self.num_dim * np.log(self.nu)
        z3 = -0.5 * (self.nu + self.num_dim) * np.log(1. + np.dot(t.T, t) / self.nu)
        z4 = np.log(special.gamma((self.nu + self.num_dim) / 2.))
        return z1 + z2 + self._const - 0.5 * self.log_det_C + z3 + z4

    def _eval_grad(self, x):
        """
        Evaluate the gradient of the log of the PDF at x.
        """
        t = scipy.linalg.solve_triangular(self.L[0], self.mu - x, lower=self.L[1])
        quadr = 1 + np.dot(t.T, t) / self.nu
        res = scipy.linalg.cho_solve(self.L, self.mu - x)[None, :]
        return (self.nu + self.num_dim) * res / (quadr * self.nu)

    def _eval_hessian(self, x):
        """
        Evaluate the Hessian of the log of the PDF at x.
        """
        t = scipy.linalg.solve_triangular(self.L[0], self.mu - x, lower=self.L[1])
        quadr = 1 + np.dot(t.T, t) / self.nu
        res = scipy.linalg.cho_solve(self.L, self.mu - x)
        return (self.nu + self.num_dim) * ( 2. * np.dot(res, np.transpose(res)) / quadr - self.inv_C ) / (quadr * self.nu)

    def _eval_grad_mu(self, x):
        """
        Evaluate the gradient with respect to mu at x.
        """
        return -self._eval_grad(x)

    def grad_mu(self, x):
        """
        Evaluate the derivative with respect to mu at x.
        """
        return call_many(x, self._eval_grad_mu)

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = super(MultivariateT, self).__str__() + '\n'
        s += 'mu:\n'
        s += str(self.mu) + '\n'
        s += 'C:\n'
        s += str(self.C) + '\n'
        s += 'nu:\n'
        s += str(self.nu)
        return s
