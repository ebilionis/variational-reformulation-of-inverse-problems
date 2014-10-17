"""
A class representing a Mixture of various pdfs .

Author:
    Panagiotis Tsilifis

Date:
    6/05/2014

"""


__all__ = ['MixturePDF']


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
#from . import MultivariateNormal
from . import regularize_array


class MixturePDF(PDFBase):

    """
    A class representing a mixture of arbitrary pdfs.

    :param comp:  A list of :class:`vuq.PDFBase`'s representing the
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

    def __init__(self, comp, w=None, name='Mixture pdf'):
        """
        Initialize the object.
        """
        assert isinstance(comp, Iterable)
        #for c in comp:
        #    assert isinstance(c, MultivariateNormal)
        num_dim = comp[0].num_dim
        for c in comp[1:]:
            assert num_dim == c.num_dim
        self._comp = comp
        super(MixturePDF, self).__init__(num_dim, name=name)
        if w is None:
            w = np.ones(self.num_comp)
        self.w = w

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
        return np.einsum('i,ijk->jk', np.exp(r - log_q), dr), dr, log_q, r

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

    def _eval_hessian(self, x):
        """
        Evaluate the Hessian of the log of the PDF at x.
        """
        g_log_q, dr, log_q, r = self._eval_grad_each(x)
        tmp = np.einsum('ijk,ikl->ijl', dr, dr)
        T = np.einsum('i,ijk->jk', np.exp(r - log_q), tmp)
        d2r = np.array([c._eval_hessian(x) for c in self.comp])
        T2 = np.einsum('i,ijk->jk', np.exp(r - log_q), d2r)
        return T2 + T - np.dot(g_log_q, g_log_q.T)

    def propagate(self, model):
        """
        Propagate the uncertainty of this mixture through ``model``.
        """
        raise NotImplementedError('Implement me!')

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = super(MixturePDF, self).__str__() + '\n'
        s += 'Num comp: ' + str(self.num_comp) + '\n'
        s += 'Weights:\n'
        s += str(self.w) + '\n'
        for i in xrange(self.num_comp):
            s += 'Comp ' + str(i) + '\n'
            s += str(self.comp[i]) + '\n'
        return s
