"""
A class that represents the log PDF of the input.

Author:
    Ilias Bilionis

Date:
    5/19/2014

"""


__all__ = ['PDFBase']


import numpy as np
from . import call_many


class PDFBase(object):

    """
    A class that represents the log PDF of the input.

    This is an abstract class that needs to be overriden by the children.
    """

    # The number of input dimensions
    _num_dim = None

    @property
    def num_dim(self):
        """
        :getter:    The number of dimensions.
        """
        return self._num_dim

    def __init__(self, num_dim, name='Input log PDF'):
        """
        Initialize the object.
        """
        self._num_dim = num_dim
        self.__name__ = name

    def _eval(self, x):
        """
        Evaluate the log PDF at a single input.

        :returns:   The log PDF at x.
        :rtype:     float
        """
        raise NotImplementedError('Re-implement me in children!')

    def _eval_grad(self, x):
        """
        Evaluate the gradient of the log PDF at a single input.

        :returns:   The gradient at x as a one row matrix with as many
                    columns as ``self.num_dim``.
        :rtype:     :class:`numpy.ndarray`
        """
        raise NotImplementedError('Re-implement me in children!')

    def _eval_hessian(self, x):
        """
        Evaluate the Hessian of the log PDF at a single input.

        :returns:   The Hessian at x as a matrix.
        :rtype:     :class:`numpy.ndarray`
        """
        raise NotImplementedError('Re-implement me in children!')

    def _sample(self):
        """
        Sample from the distribution.

        :returns:   A single sample
        """
        raise NotImplementedError('Re-implement me in children!')

    def __call__(self, x):
        """
        Evaluate the log PDF at x.

        :param x:   The input. It shoud always be a 2D matrix with the rows
                    representing different points and the columns the number
                    of inputs.
        :type x:    :class:`numpy.ndarray`
        :returns:   The log PDF evaluated at all the rows of x.
        :rtype:     :class:`numpy.ndarray`
        """
        return call_many(x, self._eval)

    def grad(self, x):
        """
        Evaluate the gradient of the log PDF at x.

        :param x:   The input. As in :meth:`vuq.Inputlog PDFBase.__call__`.
        :type x:    :class:`numpy.ndarray`
        :returns:   The gradient evaluated at all the inputs. It will be a list
                    of containing the gradient on each one of the points of rows
                    of x.
        :rtype:     list of :class:`numpy.ndarray`
        """
        return call_many(x, self._eval_grad, return_numpy=False)

    def hessian(self, x):
        """
        Evaluate the Hessian of the log PDF at x.

        :param x:   The input. As in :meth:`vuq.Inputlog PDFBase.__call__`.
        :type x:    :class:`numpy.ndarray`
        :returns:   The Hessian evaluated at all the input. It will be a list of
                    2D arrays. The 2D arrays correspond to the Hessians of each point.
                    It is allowed for these Hessians to be sparse matrices (and they
                    should).
        :rtype:     :class:`numpy.ndarray`
        """
        return call_many(x, self._eval_hessian, return_numpy=False)

    def _eval_all(self, x):
        """
        Evaluates everything and returns it as a dictionary.

        The keys of the dictionary are as follows:
        + log_p:        1D array of size x.shape[0] containing the log PDFs
        + log_p_grad:   list of size x.shape[0] containing the gradient of the PDF
                        at each row of x.
        + logp_grad_2: list of size x.shape[0] containing the hessian of the PDF
                        at each row of x.
        """
        state = {}
        state['log_p'] = self._eval(x)
        state['log_p_grad'] = self._eval_grad(x)
        state['log_p_grad_2'] = self._eval_hessian(x)
        return state

    def eval_all(self, x):
        """
        Evaluates everything at many points and returns a dictionary.
        """
        out = call_many(x, self._eval_all, return_numpy=False)
        state = {}
        state['log_p'] = np.array([s['log_p'] for s in out])
        state['log_p_grad'] = [s['log_p_grad'] for s in out]
        state['log_p_grad_2'] = [s['log_p_grad_2'] for s in out]
        return state

    def sample(self, size=1):
        """
        Sample from the distribution.

        :param size:    The number of samples to return.
        """
        return np.array([self._sample() for i in xrange(size)])

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = 'Name: ' + self.__name__ + '\n'
        s += 'Num Input: ' + str(self.num_dim)
        return s
