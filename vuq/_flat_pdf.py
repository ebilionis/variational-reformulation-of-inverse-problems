"""
A completely flat PDF.

Author:
    Ilias Bilionis

Date:
    6/16/2014

"""


__all__ = ['FlatPDF']


import numpy as np
from . import PDFBase


class FlatPDF(PDFBase):

    """
    A class representing a completely flat PDF.
    """

    def __init__(self, num_dim, name='Flat PDF'):
        """
        Initialize the object.
        """
        super(FlatPDF, self).__init__(num_dim, name=name)

    def _eval(self, x):
        """
        Evaluate the log of the PDF at x.
        """
        return 0.

    def _eval_grad(self, x):
        """
        Evaluate the gradient of the log of the PDF at x.
        """
        return np.zeros((1, self.num_dim))

    def _eval_hessian(self, x):
        """
        Evaluate the Hessian of the log of the PDF at x.
        """
        return np.zeros((self.num_dim, self.num_dim))
