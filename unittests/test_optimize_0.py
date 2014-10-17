"""
Approximate any distribution with a mixture of gaussians.

Author:
    Ilias Bilionis

Date:
    6/16/2014
    9/18/2014

"""

import numpy as np
import matplotlib.pyplot as plt
from vuq import *


# Pick a distribution:
log_p = MixtureOfMultivariateNormals.create(1, 2)
log_p.comp[0].C = [[2.5]]
log_p.comp[0].mu = [0.]
#log_p.comp[1].mu = [2.]

# Pick an approximating distribution
log_q = MixtureOfMultivariateNormals.create(1, 4)
log_q.mu[0, :] = -1.
#log_q.mu[1, :] = 2.
log_q.C[:] = 0.01
#log_q.mu = np.zeros(log_q.mu.shape) + 1e-3

# Create the elbo
entropy = EntropyLowerBound()
exp_func = ThirdOrderExpectationFunctional(log_p)
elbo = EvidenceLowerBound(entropy, exp_func)

# Optimize
optimizer = Optimizer(elbo)
#L = optimizer.optimize_full_mu(log_q)
#L = optimizer.optimize_full_L(log_q)
C_bounds = tuple((1e-6, 100) for i in xrange(log_q.num_comp * log_q.num_dim))
L = optimizer.optimize(log_q, C_bounds=C_bounds, full_mu=True, tol=1e-1)
print L

print 'Compare this:'
print log_p.mu
print 'with this:'
print log_q.mu
print 'Compare the means:'
print np.sum(log_q.w * log_q.mu.T)
print np.sum(log_p.w * log_p.mu.T)

x = np.linspace(-5, 5, 100)[:, None]
plt.plot(x, np.exp(log_p(x)), '-', linewidth=2)
plt.plot(x, np.exp(log_q(x)), '--', linewidth=2)
plt.show()
