"""
A multivariate uninformative PDF

Author:
    Panagiotis Tsilifis
Date:
    05/21/2014

"""


__all__ = ['UninformativePDF']


import numpy as np
import math
from . import view_as_column
from . import PDFBase


class UninformativePDF(PDFBase):
    """
    A class representing the PDF of an Uninformative distribution.
    """

    def __init__(self, num_dim, name='Uninformative prior'):
        """
        Initialize the object.
        """
        super(UninformativePDF, self).__init__(num_dim, name=name)

    def _eval(self, x):
        """
        Evaluate the log of the PDF at x.
        """
        return -np.sum(np.log(x))

    def _eval_grad(self, x):
        """
        Evaluate the gradient of the log of the PDF at x.
        """
        return -1/x[None, :]

    def _eval_hessian(self, x):
        """
        Evaluate the Hessian of the log of the PDF at x.
        """
        y = 1/x**2
        return np.diag(y)
