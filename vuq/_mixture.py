"""
A class representing a Mixture of Gaussians.

Author:
    Ilias Bilionis

Date:
    5/19/2014

"""


__all__ = ['MixtureOfMultivariateNormals']


import numpy as np
from scipy.misc import logsumexp
from collections import Iterable
from itertools import izip
import math
try:
    # If this fails, then you don't have sklearn
    # If this is the case, then you cannot use the
    # the function MixtureOfMultivariateNormals.fit().
    from sklearn import mixture
except:
    pass
from . import call_many
from . import PDFBase
from . import MultivariateNormal
from . import regularize_array


class MixtureOfMultivariateNormals(PDFBase):

    """
    A class representing a mixture of multivariate normals.

    :param comp:  A list of :class:`vuq.MultivariateNormal`'s representing the
                  components of the Mixture.
    :param w:     The weight of each component. If ``None`` then all components
                  receive equal weights.
    """

    # The components of the mixture
    _comp = None

    # The log of the weights of the mixture
    _log_w = None

    @property
    def comp(self):
        """
        :getter: The components of the mixture.
        """
        return self._comp

    @property
    def num_comp(self):
        """
        :getter: The number of components.
        """
        return len(self.comp)

    @property
    def w(self):
        """
        :getter: The weight of each component.
        :setter: The weight of each component.
        """
        return np.exp(self._log_w)

    @property
    def mu(self):
        """
        :getter: The mean of each component.
        :setter: The mean of each component.
        """
        return np.array([c.mu for c in self.comp])

    @mu.setter
    def mu(self, value):
        """
        Set the mean of each component.
        """
        assert isinstance(value, np.ndarray)
        value = regularize_array(value)
        assert value.shape[0] == self.num_comp
        assert value.shape[1] == self.num_dim
        for i in xrange(self.num_comp):
            self.comp[i].mu = value[i, :]

    @property
    def C(self):
        """
        :getter: The covariance of each component.
        :setter: The covariance of each component.
        """
        return np.array([c.C for c in self.comp])

    @C.setter
    def C(self, value):
        """
        Set the covariance of each component.
        """
        assert isinstance(value, np.ndarray)
        if value.ndim == 2:
            value = np.array([value])
        assert value.shape[0] == self.num_comp
        assert value.shape[1] == self.num_dim
        assert value.shape[2] == self.num_dim
        for i in xrange(self.num_comp):
            self.comp[i].C = value[i, :, :]

    @w.setter
    def w(self, value):
        """
        Set the weight of each component.
        """
        self.log_w = np.log(value)

    @property
    def log_w(self):
        """
        :getter: The logarithm of the weights.
        :setter: The logarithm of the weights.
        """
        return self._log_w

    @log_w.setter
    def log_w(self, value):
        """
        Set the logarithm of the weights.
        """
        assert isinstance(value, np.ndarray)
        assert value.ndim == 1
        assert value.shape[0] == self.num_comp
        self._log_w = value - logsumexp(value)

    def __init__(self, comp, w=None, name='Mixture of Gaussians'):
        """
        Initialize the object.
        """
        assert isinstance(comp, Iterable)
        for c in comp:
            assert isinstance(c, MultivariateNormal)
        num_dim = comp[0].num_dim
        for c in comp[1:]:
            assert num_dim == c.num_dim
        self._comp = comp
        super(MixtureOfMultivariateNormals, self).__init__(num_dim, name=name)
        if w is None:
            w = np.ones(self.num_comp)
        self.w = w

    def _sample(self):
        """
        Sample the distribution once.
        """
        i = np.arange(self.num_comp)[np.random.multinomial(1, self.w) == 1][0]
        return self.comp[i]._sample()

    def _eval_each(self, x):
        """
        Evaluate each component and return the relative log probabilities and their
        sum.
        """
        r = np.array([c._eval(x) for c in self.comp]) + self.log_w
        return logsumexp(r), r

    def _eval_grad_each(self, x):
        """
        Same as :meth:`vuq.MixtureOfMultivariateNormals._eval_each()` but for the
        gradient.
        """
        log_q, r = self._eval_each(x)
        dr = np.array([c._eval_grad(x) for c in self.comp])
        dr_C = -np.array([c._inv_C for c in self.comp])
        dr_C += np.einsum('ijk,ikj->ijk', dr, dr)
        dr_C *= 0.5
        return np.einsum('i,ijk->jk', np.exp(r - log_q), dr), dr, log_q, r, dr_C

    def _eval(self, x):
        """
        Evaluate the log of the PDF at x.
        """
        return self._eval_each(x)[0]

    def _eval_grad(self, x):
        """
        Evaluate the gradient of the log of the PDF at x.
        """
        return self._eval_grad_each(x)[0]

    def _eval_grad_x_mu_and_C(self, x):
        """
        Evaluate the gradient of the log of the PDF at x wrt x and mu.
        """
        grad_x, dr, log_q, r, dr_C = self._eval_grad_each(x)
        grad_mu = -np.einsum('i,ijk->ijk', np.exp(r - log_q), dr)
        grad_C = np.einsum('i,ijk->ijk', np.exp(r - log_q), dr_C)
        return log_q, grad_x, grad_mu, grad_C

    def _eval_hessian(self, x):
        """
        Evaluate the Hessian of the log of the PDF at x.
        """
        g_log_q, dr, log_q, r, dr_C = self._eval_grad_each(x)
        tmp = np.einsum('ijk,ikl->ijl', dr, dr)
        T = np.einsum('i,ijk->jk', np.exp(r - log_q), tmp)
        d2r = np.array([c._eval_hessian(x) for c in self.comp])
        T2 = np.einsum('i,ijk->jk', np.exp(r - log_q), d2r)
        return T2 + T - np.dot(g_log_q, g_log_q.T)

    def log_q_grad_x_mu_and_C(self, x):
        """
        Evaluate the log of the PDF at x and its  gradient wrt x and mu.
        """
        tmp = call_many(x, self._eval_grad_x_mu_and_C, return_numpy=False)
        log_q = np.array([o[0] for o in tmp])
        grad_x = np.array([o[1] for o in tmp])
        grad_mu = np.array([o[2] for o in tmp])
        grad_C = np.array([o[3] for o in tmp])
        return log_q, grad_x, grad_mu, grad_C

    @staticmethod
    def create(num_dim, num_comp):
        """
        Create a mixture of Gaussians with ``num_dim`` dimensions, random centers
        and unit covariance matrices, and equal weights.

        :param num_dim: Number of dimensions.
        :type num_dim:  int
        :param num_comp: Number of components.
        :type num_comp: int
        """
        comp = [MultivariateNormal(np.random.randn(num_dim))
                for i in xrange(num_comp)]
        return MixtureOfMultivariateNormals(comp)

    @staticmethod
    def fit(X, num_comp):
        """
        Fit the data ``X`` to a mixture of Gaussians using ``num_comp`` components.

        It uses the same covariance for each one.

        :param X:   The data.
        :type X:    :class:`numpy.ndarray`
        :param num_comp:    The number of components.
        :type num_comp:     int
        """
        gmm = mixture.GMM(n_components=num_comp, covariance_type='tied')
        gmm.fit(input_samples)
        mu = gmm.means_
        C = gmm.covars_
        w = gmm.weights_
        comp = [MultivariateNormal(mu[i, :], C=C) for i in xrange(gmm.n_components)]
        return MixtureOfMultivariateNormals(comp, w=w)

    def propagate(self, model):
        """
        Propagate the uncertainty of this mixture through ``model``.
        """
        raise NotImplementedError('Implement me!')

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = super(MixtureOfMultivariateNormals, self).__str__() + '\n'
        s += 'Num comp: ' + str(self.num_comp) + '\n'
        s += 'Weights:\n'
        s += str(self.w) + '\n'
        for i in xrange(self.num_comp):
            s += 'Comp ' + str(i) + '\n'
            s += str(self.comp[i]) + '\n'
        return s
