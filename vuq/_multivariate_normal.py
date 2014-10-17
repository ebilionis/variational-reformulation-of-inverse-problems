"""
A multivariate normal PDF.

Author:
    Ilias Bilionis

Date:
    5/19/2014

"""


__all__ = ['MultivariateNormal']


import numpy as np
import math
import scipy.linalg
from . import make_vector
from . import call_many
from . import PDFBase


class MultivariateNormal(PDFBase):

    """
    A class representing the PDF of a multivariate Normal distribution.

    :param mu:  The mean of the distribution.
    :type mu:   :class:`numpy.ndarray`
    :param C:   The covariance matrix. It is taken to be the unit matrix,
                if it is not specified.
    :type C:    :class:`numpy.ndarray`

    """

    # The mean
    _mu = None

    # The covariance
    _C = None

    # The Cholesky decomposition of C
    _L = None

    # The log of determinant of C
    _log_det_C = None

    # The inverse of the covariance matrix
    _inv_C = None

    @property
    def mu(self):
        """
        :getter:    The mean of the distribution. Internally, it is represented
                    as a row matrix.
        :setter:    Set the mean.
        """
        return self._mu

    @mu.setter
    def mu(self, value):
        """
        Set the mean.
        """
        value = make_vector(value)
        assert value.shape[0] == self.num_dim
        self._mu = value

    @property
    def C(self):
        """
        :getter:    The covariance matrix.
        :setter:    Set the covariance matrix.
        """
        return self._C

    @C.setter
    def C(self, value):
        """
        Set the covariance matrix
        """
        value = np.array(value)
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
    def entropy(self):
        """
        :getter:    The entropy of the distribution.
        """
        return 0.5 * self.num_dim * (1. + math.log(2. * math.pi)) + 0.5 * self.log_det_C

    def __init__(self, mu, C=None, name='Multivariate Normal'):
        """
        Initialize the object.
        """
        self._mu = make_vector(mu)
        super(MultivariateNormal, self).__init__(self.mu.shape[0], name=name)
        if C is None:
            C = np.eye(self.num_dim)
        self.C = C
        # const is the part of the likelihood that does not depend on any parameter
        self._const = -0.5 * self.num_dim * math.log(2. * math.pi)

    def _sample(self):
        """
        Return a single sample.
        """
        z = np.random.randn(self.L[0].shape[1])
        y = np.dot(self.L[0], z)
        return self.mu.flatten() + y

    def _eval(self, x):
        """
        Evaluate the log of the PDF at x.
        """
        t = scipy.linalg.solve_triangular(self.L[0], self.mu - x, lower=self.L[1])
        return self._const - 0.5 * self.log_det_C - 0.5 * np.dot(t.T, t)

    def _eval_grad(self, x):
        """
        Evaluate the gradient of the log of the PDF at x.
        """
        res = scipy.linalg.cho_solve(self.L, self.mu - x)
        return res[None, :]

    def _eval_hessian(self, x):
        """
        Evaluate the Hessian of the log of the PDF at x.
        """
        return -self.inv_C

    def _eval_grad_mu(self, x):
        """
        Evaluate the gradient with respect to mu at x.
        """
        return -self._eval_grad(x)

    def _eval_grad_C(self, x):
        """
        Evaluate the gradient with respect to C at x.
        """
        grad_x = self._eval_grad(x)
        res = np.einsum('ji,jk->ik', grad_x, grad_x)
        res -= self._inv_C
        res *= 0.5
        return res

    def grad_mu(self, x):
        """
        Evaluate the derivative with respect to mu at x.
        """
        return call_many(x, self._eval_grad_mu)

    def grad_C(self, x):
        """
        Evaluate the derivative with respect to C at x.
        """
        return call_many(x, self._eval_grad_C)

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = super(MultivariateNormal, self).__str__() + '\n'
        s += 'mu:\n'
        s += str(self.mu) + '\n'
        s += 'C:\n'
        s += str(self.C)
        return s
