"""
A multivariate Uniform PDF. 

Author:
    Panagiotis Tsilifis
Date:
    06/16/2014

"""


__all__ = ['UniformND']


import numpy as np
from . import view_as_column
from . import PDFBase


class UniformND(PDFBase):
    """
    A class representing the n-d Uniform PDF on [0,1]^n.
    """
    
    def __init__(self, n, name='Multivariate Uniform'):
        """
        Initialize the object.
        """
        super(UniformND, self).__init__(n, name=name)
    
    def _eval(self, x):
        """
        Evaluate the log of the PDF at x.
        """
        return 0.
    
    def _eval_grad(self, x):
        """
        Evaluate the gradient of the log of the pdf at x.
        """
        x = view_as_column(x)
        assert x.shape[0] == self.num_dim
        return np.zeros(x.shape[0])[None, :]
    
    def _eval_hessian(self, x):
        """
        Evaluate the Hessian of the log of the pdf at x.
        """
        x = view_as_column(x)
        assert x.shape[0] == self.num_dim
        return np.zeros((x.shape[0],x.shape[0]))

    def _sample(self):
        """
        Sample from the distribution.
        """
        return np.random.rand(self.num_dim)
