"""
A class representing the expecation of a log PDF over a mixture.

Author:
    Ilias Bilionis

Date:
    6/5/2014

"""


__all__ = ['ExpectationFunctional']


from . import PDFBase
from . import Joint


class ExpectationFunctional(object):

    """
    A class representing the expectation of a log PDF over a mixture:

    .. math::

        \mathcal{F}[q] = \int q(\omega) \ln p(\omega) d\omega.

    """

    # A name for the object
    __name__ = None

    # The underlying log_p
    _log_p = None

    @property
    def log_p(self):
        """
        :getter:    The underlying log PDF
        :setter:    The underlying log PDF
        """
        return self._log_p

    @log_p.setter
    def log_p(self, value):
        """
        Set the underlying log PDF.
        """
        assert isinstance(value, PDFBase) or isinstance(value, Joint)
        self._log_p = value

    def __init__(self, log_p, name='Expectation Functional'):
        """
        Initialize the object.
        """
        self.log_p = log_p
        self.__name__ = name

    def __call__(self, q):
        """
        Evaluate the functional at ``q`` and the derivatives with respect to
        all parameters.
        """
        raise NotImplementedError('Implement me!')

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = 'Name: ' + self.__name__ + '\n'
        s += 'Underlying log PDF:\n'
        s += str(self.log_p)
        return s
