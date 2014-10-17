"""
A class that represents a collection of PDF's.

Author:
    Ilias Bilionis

Date:
    5/22/2014

"""


__all__ = ['PDFCollection']


from collections import Iterable
from itertools import izip
import numpy as np
from scipy.sparse import block_diag
from . import PDFBase


class PDFCollection(PDFBase):
    
    """
    This is a class that can be used to represent the PDF of many
    random variables that are independent and are made out of other PDF's.

    The problem it solves is this:
    + Suppose that you have n PDFBase objects p1, ..., pn corresponding to indepdenent
      random variables x1, ..., xn and that you want to make a single random variable x
      so that x = (x1, ..., xn).
    Well, this class does exactly that. It represents this collection in a unified way.
    """

    # The collection of random variables
    _collection = None

    # A list as big as the number of components in the collection
    # so that _last_idx[i] corresponds to the last index of the input
    # pertaining to the ith component.
    _idx = None

    @property
    def collection(self):
        """
        :getter:    The collection of random variables.
        """
        return self._collection

    @property
    def num_comp(self):
        """
        :getter:    The number of components.
        """
        return len(self.collection)

    def _extract_part(self, x, i):
        """
        Extract the part of the input that pertains to the ith component.
        """
        first_idx = 0 if i == 0 else self._idx[i - 1]
        last_idx = self.num_dim if i == self.num_comp - 1 else self._idx[i]
        return x[first_idx:last_idx]

    def _extract_parts(self, x):
        """
        Extracts the parts of the input pertaining to each component.
        """
        return [self._extract_part(x, i)
                for i in xrange(self.num_comp)]

    def __init__(self, collection, name='PDF Collection'):
        """
        Initialize the object.
        """
        assert isinstance(collection, Iterable)
        for p in collection:
            assert isinstance(p, PDFBase)
        self._collection = collection
        dim = [p.num_dim for p in collection]
        self._idx = np.cumsum(dim)
        num_dim = np.sum([p.num_dim for p in collection])
        super(PDFCollection, self).__init__(num_dim, name=name)

    def _meta_eval(self, x, func):
        """
        Evaluate each function ``func`` of each component on the relevant part.
        """
        x_parts = self._extract_parts(x)
        return [getattr(c, func)(x_p)
                for x_p, c in izip(x_parts, self.collection)]

    def _eval(self, x):
        """
        Evaluate the log of the PDF at x.
        """
        return np.sum(self._meta_eval(x, '_eval'))

    def _eval_grad(self, x):
        """
        Evaluate the gradient of the log of the PDF at x.
        """
        return np.hstack(self._meta_eval(x, '_eval_grad'))

    def _eval_hessian(self, x):
        """
        Evaluate the Hessian of the log of the PDF at x.
        It is a sparse matrix.
        """
        y = self._meta_eval(x, '_eval_hessian')
        return block_diag(self._meta_eval(x, '_eval_hessian'), format='lil')

    def _sample(self):
        """
        Return a single sample.
        """
        return np.hstack([p._sample() for p in self._collection])

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = super(PDFCollection, self).__str__() + '\n'
        for i in xrange(self.num_comp):
            s += 'Component: ' + str(i) + '\n'
            s += str(self.collection[i]) + '\n'
        return s
