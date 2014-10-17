"""
A class representing an approximation to the entropy.

Author:
    Ilias Bilionis

Date:
    6/3/2014

"""


__all__ = ['EntropyApproximation']


class EntropyApproximation(object):

    """
    A class representing an entropy approximation.

    In everything that follows ``q`` is assumed to be a
    :class:`vuq.MixtureOfMultivariateNormals`. We do not know how to evaluate
    the entropy of any other distribution.

    The following conventions are followed:
    + ``w`` is the vector of responsibilities of ``q``.
    + ``mu`` are the means of each component of ``q``.
    + ``C`` are the covariances of ``q``.
    """

    # A name for the object
    __name__ = None

    def __init__(self, name='Entropy Approximation'):
        """
        Initialize the model.
        """
        self.__name__ = name

    def __str__(self):
        """
        Return a string representation of the object.
        """
        return 'Name: ' + self.__name__

    def eval_all_for_mu(self, i, q):
        """
        Evaluate the entropy along with the gradient and the Hessian with respect
        to the mu of the i-th component of q.
        """
        raise NotImplementedError('Implement me!')

    def __call__(self, q):
        """
        Evaluate the entropy functional at ``q``.
        """
        raise NotImplementedError('Implement me!')
