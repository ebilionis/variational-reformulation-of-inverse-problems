"""
Implements a class that represents the joint distribution (likelihood x prior).

Author:
    Ilias Bilionis

Date:
    5/23/2014

"""


__all__ = ['Joint']


import numpy as np
from . import regularize_array
from . import view_as_column
from . import call_many
from . import Likelihood
from . import PDFBase


class Joint(PDFBase):

    """
    A class that represents the joint distribution.
    """

    # The likelihood function
    _likelihood = None

    # The prior pdf
    _prior = None

    @property
    def likelihood(self):
        """
        :getter:    The likelihood function.
        """
        return self._likelihood

    @property
    def prior(self):
        """
        :getter:    The prior function.
        """
        return self._prior

    @property
    def num_params(self):
        """
        :getter:    The number of parameters.
        """
        return self.likelihood.num_params

    def __init__(self, likelihood, prior):
        """
        Initialize the object.
        """
        assert isinstance(likelihood, Likelihood)
        assert isinstance(prior, PDFBase)
        assert likelihood.num_params == prior.num_dim
        self._likelihood = likelihood
        self._prior = prior
        super(Joint, self).__init__(prior.num_dim, name='Joint PDF')

    def _eval(self, omega):
        """
        TODO: Make more efficient by re-defining the likelihood.
        """
        return self._eval_all(omega)['log_p']
    
    def _eval_all(self, omega):
        """
        Evaluate everything and put it in a dictionary.
        """
        l_state = self.likelihood._eval(omega)
        p_state = self.prior._eval_all(omega)
        state = {}
        state['log_p'] = l_state['L'] + p_state['log_p']
        state['log_p_grad'] = l_state['L_grad'] + p_state['log_p_grad']
        state['log_p_grad_2'] = l_state['L_grad_2'] + p_state['log_p_grad_2']
        return state

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = 'Log of Joint distribution.\n'
        s += 'Underlying likelihood:\n'
        s += str(self.likelihood) + '\n'
        s += 'Underlying prior:\n'
        s += str(self.prior)
        return s
