"""
Implements a class representing a likelihood function.

Author:
    Ilias Bilionis

Date:
    5/22/2014

"""


__all__ = ['Likelihood']


import numpy as np
from . import regularize_array
from . import call_many
from . import CachedFunction
from . import Model
from . import PDFBase


class Likelihood(object):

    """
    A class that represents a log likelihood function.

    This is an abstract class that needs to be overriden by the children.

    A log likelihood function is a function like this:

            L(y, f(x), theta)           (1)

    where f is a vuq.Model, x are the inputs of the model and theta are the parameters
    of the likelihood function.

    We will be denoting collectively all the parameters by:

            omega = (x, theta)          (2)

    The goal of this class is to make it easy to evaluate L, as well as its first
    and second derivatives.

    """

    # A name for the likelihood
    __name__ = None

    # The observed data
    _y = None

    # The model
    _model = None

    # The number of dimensions (of the input of the model)
    _num_dim = None

    # The number of parameters of the likelihood function only
    _num_like_params = None

    # The noise model
    _noise = None

    @property
    def model(self):
        """
        :getter:    The underlying model.
        """
        return self._model

    @property
    def num_like_params(self):
        """
        :getter:    The number of likelihood parameters.
        """
        return self._num_like_params

    @property
    def num_params(self):
        """
        :getter:    The number of parameters.
        """
        if self.model is not None:
            return self.model.num_input + self.num_like_params
        else:
            return self.num_dim + self.num_like_params

    @property
    def num_data(self):
        """
        :getter:    The dimensionality of the data accepted by this likelihood
                    function.
        """
        return self._model.num_output

    @property
    def y(self):
        """
        :getter:    The observed data.
        """
        return self._y

    def __init__(self, y, model=None, num_like_params=None,
                 num_dim=None, name='Likelihood'):
        """
        Initialize the model.
        """
        self._y = y
        if model is None and num_dim is None:
            raise RuntimeError('One of the ``model`` or ``num_dim`` must be defined!')
        if model is not None:
            if num_dim is not None:
                raise RuntimeError('Exactly one of the ``model`` or ``num_dim`` must be defined!')
            assert isinstance(model, Model)
            self._model = model
        elif num_dim is not None:
            self._num_dim = num_dim
        self._num_like_params = int(num_like_params)
        self.__name__ = str(name)
        self._eval = CachedFunction(self.__eval)
        self._model_eval = CachedFunction(self.model._eval)

    def _noise_eval(self, fx, theta):
        """
        Evaluate the log likelihood (and 1st and 2nd derivatives) when the underlying
        model is ``fx`` and the likelihood's parameters are ``theta``.

        That is it should evaluate this function:

                L(y, fx, theta)

        ``y`` is not given as an input. You can find it by looking at ``self.y``.

        :returns:   A dictionary with the following keys:
                    + L:            The evaluation of L(y, fx, theta), float
                    + L_grad_f:     The Jacobian of L with respect to f, 2D array
                                    of size 1 x self.model.num_output.
                    + L_grad_theta: The Jacobian of L with respect to theta, 2D array
                                    of size 1 x self.num_like_params.
                    + L_grad_2_f:   The Hessian of L with respect to f, 2D array
                                    of size self.model.num_output x self.model.num_output
                    + L_grad_2_theta: The Hessian of L with respect to theta, 2D array
                                    of size self.num_like_params x self.num_like_params.
                                    If there are no likelihood prameters, this output
                                    is ignored.
        """
        raise NotImplementedError('My children should implement this!')

    def __eval(self, omega):
        """
        Evaluate the likelihood at a single parameter ``omega``.

        :returns:   A dictionary with the following keys:
                    + L:        The log likelihood at omega.
                    + L_grad:   The gradient at omega, 2D array of 1 x omega.shape[0]
                                dimesnions.
                    + L_grad_2: The Hessian at omega, 2D array of
                                omega.shape[0] x omega.shape[0] dimensions.
        """
        x = omega[:self.model.num_input]
        theta = omega[self.model.num_input:]
        # Sanity check
        assert theta.shape[0] == self.num_like_params
        # Evaluate the model
        m_state = self._model_eval(x)
        f = m_state['f']
        f_grad_x = m_state['f_grad']
        f_grad_2_x = m_state['f_grad_2']
        # Evaluate the noise
        l_state = self._noise_eval(f, theta)
        L = l_state['L']
        L_grad_f = l_state['L_grad_f']
        L_grad_theta = l_state['L_grad_theta']
        L_grad_2_f = l_state['L_grad_2_f']
        L_grad_2_theta = l_state['L_grad_2_theta']
        L_grad_2_theta_f = l_state['L_grad_2_theta_f']
        # Now use the chain rule to evaluate the Jacobian
        L_grad_x = np.einsum('ij,jk->ik', L_grad_f, f_grad_x)
        L_grad_omega = np.hstack([L_grad_x, L_grad_theta])
        # Finally, use the chain rule to compute the second derivative
        L_grad_2_x = np.einsum('ij,ik,jl->kl', L_grad_2_f, f_grad_x, f_grad_x)
        # TODO: Make it work with f_grad_2_x being a list of 2D arrays
        L_grad_2_x += np.einsum('ijk,li->ljk', f_grad_2_x, L_grad_f)[0, :, :]
        L_grad_2_theta_x = np.einsum('ij,jk->ik', L_grad_2_theta_f, f_grad_x)
        # END OF TODO.
        L_grad_2_omega = np.bmat([[L_grad_2_x, L_grad_2_theta_x.T],
                               [L_grad_2_theta_x, L_grad_2_theta]])
        state = {}
        state['L'] = L
        state['L_grad'] = L_grad_omega
        state['L_grad_2'] = L_grad_2_omega
        state['m_state'] = m_state
        state['l_state'] = l_state
        return state

    def __call__(self, omega):
        """
        Evaluate the log likelihood at many inputs ``x`` when the data is ``y``.

        :param omega:   The parameters. This is not only the parameters of the underlying
                        model, but also any parameters that the noise model represented
                        by this likelihood function has (the number of the likelihood
                        parameters could be just zero).
        :type omega:   :class:`numpy.ndarray`
        :returns:       A dictionary with the following keys:
                        + L:        The log likelihoods one on rows of omega.
                        + L_grad:   List of gradients of all rows of omega.
                        + L_grad_2: List of Hessians of all rows of omega.
        """
        out = call_many(omega, self._eval, return_numpy=False)
        state = {}
        state['L'] = np.array([s['L'] for s in out])
        state['L_grad'] = [s['L_grad'] for s in out]
        state['L_grad_2'] = [s['L_grad_2'] for s in out]
        return state

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = 'Name: ' + self.__name__ + '\n'
        s += 'Total number of parameters: ' + str(self.num_params) + '\n'
        s += str(self._eval) + '\n'
        s += 'Underlying model:\n'
        s += str(self.model)
        s += str(self._model_eval)
        return s

    def freeze(self, theta):
        """
        Freeze the likelihood to have a specific theta.

        A new likelihood object is returned with the desired property.
        """
        return _FixedThetaLikelihood(self, theta)


class _FixedThetaLikelihood(Likelihood):

    """
    A likelihood with a fixed theta.

    It is a hidden class.
    """

    # The underlying likelihood
    _like = None

    # The fixed theta
    _theta = None

    @property
    def theta(self):
        """
        :getter:    Theta
        :setter:    Theta
        """
        return self._theta

    @theta.setter
    def theta(self, value):
        """
        Set the theta.
        """
        value = np.array(value)
        assert value.ndim == 1
        assert value.shape[0] == self._like.num_like_params
        self._theta = value

    def __init__(self, like, theta):
        """
        Initialize the object.
        """
        assert isinstance(like, Likelihood)
        self._like = like
        self.theta = theta
        super(_FixedThetaLikelihood, self).__init__(like.y,
                                                    like.model,
                                                    0,
                                                    like._num_dim,
                                                    name='Fixed Theta ' + like.__name__)

    def _eval(self, omega):
        """
        Just evaluate the underlying likelihood at the fixed theta.
        """
        state = self._like._eval(np.hstack([omega, self.theta]))
        state['L_grad_theta'] = [[]]
        state['L_grad_2_theta'] = [[]]
        return state

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = super(_FixedThetaLikelihood, self).__str__() + '\n'
        s += 'Theta:\n'
        s += str(self.theta)
        return s
