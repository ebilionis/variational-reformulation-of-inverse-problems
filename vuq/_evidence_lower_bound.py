"""
A class representing the lower bound to the evidence.

Author:
    Ilias Bilionis

Date:
    6/5/2014

"""


__all__ = ['EvidenceLowerBound']


from . import EntropyApproximation
from . import ExpectationFunctional
from . import MixtureOfMultivariateNormals
from . import FirstOrderEntropyApproximation
from . import ThirdOrderExpectationFunctional


class EvidenceLowerBound(object):

    """
    A class representing the lower bound to the evidence (or the negative
    KL divergence).

    :param S:   An entropy approximation.
    :type S:    :class:`vuq.EntropyApproximation`
    :param E:   An expectation functional.
    :type E:    :class:`vuq.ExpectationFunctional

    """

    # A name for the object
    __name__ = None

    # The underlying entropy object
    _entropy = None

    # The underlying expectation functional
    _expectation_functional = None

    @property
    def entropy(self):
        """
        :getter:    The entropy approximation.
        """
        return self._entropy

    @property
    def expectation_functional(self):
        """
        :getter:    The expectation functional.
        """
        return self._expectation_functional

    def __init__(self, entropy, expectation_functional,
                 name='Evidence Lower Bound'):
        """
        Initialize the object.
        """
        self.__name__ = name
        assert isinstance(entropy, EntropyApproximation)
        assert isinstance(expectation_functional, ExpectationFunctional)
        self._entropy = entropy
        self._expectation_functional = expectation_functional

    def __call__(self, q):
        """
        Evaluate the bound at `q` (and all derivatives).
        """
        S = self.entropy(q)
        F = self.expectation_functional(q)
        state = {}
        state['L'] = S['S'] + F['F']
        state['L1'] = S['S'] + F['F1']
        state['L_grad_w'] = S['S_grad_w'] + F['F_grad_w']
        state['L_grad_mu'] = S['S_grad_mu'] + F['F_grad_mu']
        state['L_grad_C'] = S['S_grad_C'] + F['F_grad_C']
        state['S_state'] = S
        state['F_state'] = F
        return state

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = 'Name: ' + self.__name__ + '\n'
        s += 'Entropy:\n'
        s += str(self.entropy)
        s += 'Expectation Functional:\n'
        s += str(self.expectation_functional)
        return s

    @staticmethod
    def create(log_p):
        entropy = FirstOrderEntropyApproximation()
        expectation_functional = ThirdOrderExpectationFunctional(log_p)
        return EvidenceLowerBound(entropy, expectation_functional)
