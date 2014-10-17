"""
A Monte Carlo approximation to the entropy.

This approximation should be assymptotically exact.

Author:
    Ilias Bilionis

Date:
    6/3/2014

"""


__all__ = ['MonteCarloEntropyApproximation']


import numpy as np
from . import EntropyApproximation
from . import PDFBase


class MonteCarloEntropyApproximation(EntropyApproximation):

    """
    A Monte Carlo enropy approximation.

    See :class:`vuq.EntropyApproximation` for more documentation.
    """

    # The number of samples to use in the approximation
    _num_samples = None

    @property
    def num_samples(self):
        """
        :getter: The number of samples.
        :setter: The number of samples.
        """
        return self._num_samples

    @num_samples.setter
    def num_samples(self, value):
        """
        Set the number of samples.
        """
        assert isinstance(value, int)
        assert value > 0
        self._num_samples = value

    def __init__(self, num_samples=100,
                 name='Monte Carlo Entropy Approximation'):
        """
        Initialize the object.
        """
        super(MonteCarloEntropyApproximation, self).__init__(name=name)
        self.num_samples = num_samples

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = super(MonteCarloEntropyApproximation, self).__str__() + '\n'
        s += 'num samples: ' + str(self.num_samples)
        return s

    def __call__(self, q):
        """
        Evaluate the likelihood.
        """
        assert isinstance(q, PDFBase)
        # Sample q many times
        x = q.sample(size=self.num_samples)
        log_q = q(x)
        return -np.mean(log_q)
