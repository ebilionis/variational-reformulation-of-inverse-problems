"""
Test the multivariate normal class.

Author:
    Ilias Bilionis

Date:
    5/19/2014

"""


import numpy as np
import scipy.stats as st
from vuq import MultivariateNormal


def dn_mu(log_p, x, h=1e-6):
    """
    Evaluate the numerical derivatives wrt mu at x.
    """
    mu = np.copy(log_p.mu)
    mu_h = np.zeros(mu.shape)
    J = np.zeros((x.shape[0], 1, log_p.num_dim))
    log_p0 = log_p(x)
    for j in xrange(log_p.num_dim):
        mu_h[:] = mu
        mu_h[j] += h
        log_p.mu = mu_h
        log_p1 = log_p(x)
        J[:, 0, j] = (log_p1 - log_p0) / h
        log_p.mu = mu
    return J


def dn_C(log_p, x, h=1e-6):
    """
    Evaluate the numerical derivatives wrt C at x.
    """
    C = np.copy(log_p.C)
    C_h = np.zeros(C.shape)
    J = np.zeros((x.shape[0], C.shape[0], C.shape[1]))
    log_p0 = log_p(x)
    print log_p0[0]
    for j in xrange(log_p.num_dim):
        for k in xrange(log_p.num_dim):
            C_h[:] = C
            C_h[j, k] += h
            if k != j:
                C_h[k, j] += h
            log_p.C = C_h
            log_p1 = log_p(x)
            J[:, j, k] = (log_p1 - log_p0) / h
            if k != j:
                J[:, j, k] *= 0.5
            log_p.C = C
    return J


# The number of dimensions
num_input = 2

# The mean
mu = np.random.randn(num_input)

# The input pdf
log_p = MultivariateNormal(mu)

# Print the object
print str(log_p) + '\n'

# Evaluate the distribution at some random points
# and compare to what we get from numpy
print 'Our Normal\t\tScipy Normal\t\tAbs. Difference'
print '-' * 80
for i in xrange(10):
    x = np.random.randn(num_input)
    sval = np.sum([st.norm.logpdf(x[i], loc=mu[i]) for i in xrange(num_input)])
    val = log_p._eval(x)
    print '{0:7f}\t\t{1:7f}\t\t{2:7f}'.format(val, sval, np.abs(val - sval))
print '-' * 80 + '\n'

# Test the evaluation at many inputs simultaneously
print 'Evaluating at many points'
print '-' * 80
x = np.random.randn(4, num_input)
y = log_p(x)
print 'Shape:', y.shape
print 'log pdf:'
print str(y) + '\n'

# Test the evaluation of the gradient at many inputs simultaneously
print 'Evaluating gradient at many points'
print '-' * 80
dy = log_p.grad(x)
print 'Shape:', len(dy), 'x', dy[0].shape
print 'grad log pdf:'
print str(dy) + '\n'

# Test the evaluation of the Hessian at many inputs simultaneously
print 'Evaluating the Hessian at many points'
print '-' * 80
hy = log_p.hessian(x)
print 'Shape:', len(hy), 'x', hy[0].shape

# Now let us test the derivatives with respect to mu and C
# and compare them to numerical derivatives
grad_mu = log_p.grad_mu(x)
n_grad_mu = dn_mu(log_p, x)
print 'grad mu:'
print grad_mu
print 'n grad mu:'
print n_grad_mu
print 'grad diff:'
print grad_mu - n_grad_mu
grad_C = log_p.grad_C(x)
n_grad_C = dn_C(log_p, x)
print 'grad C:'
print grad_C
print 'n grad C:'
print n_grad_C
print 'grad diff:'
print grad_C - n_grad_C
