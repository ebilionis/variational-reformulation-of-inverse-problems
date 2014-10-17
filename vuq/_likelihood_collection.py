"""
This is a likelihood defined on a big output that consists of independent
parts that act on disjoint parts of the output.

Author:
    Ilias Bilionis

Date:
    6/10/2014

"""


__all__ = ['LikelihoodCollection']


from collections import Iterable
from itertools import izip
import numpy as np
from scipy.sparse import block_diag
from . import Model
from . import Likelihood


class LikelihoodCollection(Likelihood):

    """
    A likelihood class that allows the application of different independent
    likelihoods on the output vector.
    """

    # The underlying likelihoods
    _collection = None

    # A list as big as the number of components in the collection
    # so that _last_idx[i] corresponds to the last index of the input
    # pertaining to the ith component.
    _idx_dim = None

    _idx_like_params = None

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

    def _extract_part(self, x, i, idx, num_dim):
        """
        Extract the part of the input that pertains to the ith component.
        """
        first_idx = 0 if i == 0 else idx[i - 1]
        idx = num_dim if i == self.num_comp - 1 else idx[i]
        return x[first_idx:last_idx]

    def _extract_parts(self, x, idx, num_dim):
        """
        Extracts the parts of the input pertaining to each component.
        """
        return [self._extract_part(x, i, idx, num_dim)
                for i in xrange(self.num_comp)]

    def __init__(self, model, collection, name='Likelihood Collection'):
        """
        Initialize the object.
        """
        assert isinstance(collection, Iterable)
        for l in collection:
            assert isinstance(l, Likelihood)
            assert hasattr(l, '_num_dim')
        dim = [l._num_dim for l in collection]
        self._idx_dim = np.cumsum(dim)
        like_params = [l.num_like_params for l in collection]
        self._idx_like_params = np.cumsum(like_params)
        num_dim = np.sum(dim)
        num_like_params = np.sum(like_params)
        super(Likelihood, self).__init__(model=model,
                                         num_like_params=num_like_params,
                                         name=name)

    def _noise_eval(self, fx, theta):
        """
        Evaluate the noise.
        """
        fx_part = self._extract_parts(fx, self._idx_dim, self.model.num_input)
        theta_part = self._extract_parts(theta, self._idx_like_params, self.num_like_params)
        states = [c._noise_eval(f, t) for f, t in izip(fx_part, theta_part)]
        state = {}
        state['L'] = np.sum([s['L'] for s in states])
        state['L_grad_f'] = np.hstack([s['L_grad_f'] for s in states])
        state['L_grad_theta'] = np.hstack([s['L_grad_theta'] for s in states])
        state['L_grad_2_f'] = block_diag([s['L_grad_2_f'] for s in states])
        state['L_grad_2_theta'] = block_diag([s['L_grad_2_theta'] for s in states])
        state['L_grad_2_theta_f'] = block_diag([s['L_grad_2_theta_f'] for s in states])
        return state
