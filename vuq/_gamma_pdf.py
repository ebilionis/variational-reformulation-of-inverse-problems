"""
A multivariate Gamma PDF (product of independent Gammas)

Author:
    Panagiotis Tsilifis
Date:
    05/21/2014

"""


__all__ = ['GammaPDF']


import numpy as np
import math
from scipy import special
from . import view_as_column
from . import PDFBase


class GammaPDF(PDFBase):
    """
    A class representing the joint PDF of independent Gamma distributions.
    """

    # Shape
    _alpha = None
    
    # Scale 
    _beta = None 
    
    @property
    def alpha(self):

        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        """
        Set parameter alpha
        """
        assert value > 0
        self._alpha = value
    
    @property
    def beta(self):
        
        return self._beta
    
    @beta.setter
    def beta(self, value):
        """
        Set parameter beta
        """
        assert value > 0
        self._beta = value
    
    def __init__(self, alpha, beta, num_dim, name='Independent Gammas'):
        """
        Initialize the object.
        """
        self._alpha = alpha
        self._beta = beta
        super(GammaPDF, self).__init__(num_dim, name=name)
    
    def _eval(self, x):
        """
        Evaluate the log of the PDF at x.
        """
        x = view_as_column(x)
        if (x <= 0.).any():
            return -np.inf
        assert x.shape[0] == self.num_dim
        z1 = self.num_dim*np.log(special.gamma(self.alpha))
        z2 = self.num_dim*self._alpha*np.log(self.beta)
        return - z1 - z2 + (self.alpha - 1)*sum(np.log(x)) - sum(x)/self.beta
    
    def _eval_grad(self, x):
        """
        Evaluate the gradient of the log of the pdf at x.
        """
        x = view_as_column(x)
        assert x.shape[0] == self.num_dim
        return (self.alpha - 1.)*(1./x) - 1/self.beta
    
    def _eval_hessian(self, x):
        """
        Evaluate the Hessian of the log of the pdf at x.
        """
        x = view_as_column(x)
        assert x.shape[0] == self.num_dim
        y = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            y[i] = -(self.alpha - 1)/x[i]**2
        return np.diag(y)

    def _sample(self):
        """
        Return a single sample.
        """
        return np.random.gamma(shape=self.alpha, scale=self.beta, size=(self.num_dim, ))
