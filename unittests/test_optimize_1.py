"""
A very simple test for the ELBO.

The target is now a mixture of two Gaussians.

The approximating mixture has two components.

Author:
    Ilias Bilionis

Date:
    6/5/2014
    9/18/2014

"""


import numpy as np
import matplotlib.pyplot as plt
import os
from vuq import *


# Number of dimensions
num_dim = 1

# The number of components to use for the mixture
num_comp = 2

# The (hypothetical) joint distribution
log_p = MixtureOfMultivariateNormals.create(num_dim, 2)
log_p.comp[0].mu = [-1.]
log_p.comp[0].C = [[0.8]]

log_p.comp[1].mu = [1.]
log_p.comp[1].C = [[0.8]]
print 'Target:'
print str(log_p)

# The approximating distribution
log_q = MixtureOfMultivariateNormals.create(num_dim, num_comp)
print 'Initial:'
print log_q
#log_q.comp[0].mu = [-5.]
#log_q.comp[1].mu = [5.]

# Evaluate the original approximation (for a plot at the end)
x = np.linspace(-5, 5, 100)[:, None]
log_q_0 = log_q(x)

# Pick an entropy approximation
entropy = EntropyLowerBound()
exp_func = ThirdOrderExpectationFunctional(log_p)
elbo = EvidenceLowerBound(entropy, exp_func)
print 'ELBO:'
print str(elbo)

# Optimize the elbo
optimizer = Optimizer(elbo)
L = optimizer.optimize(log_q, full_mu=True, tol=1e-6)[0]

# Plot the elbo
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(L, linewidth=2)
ax.set_xlabel('Iteration', fontsize=16)
ax.set_ylabel('ELBO', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
png_file = os.path.join('figures', 'optimize_1_elbo.png')
print 'writing:', png_file
plt.savefig(png_file)

# Let's plot the distributions
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, np.exp(log_p(x)), linewidth=2)
ax.plot(x, np.exp(log_q_0), linewidth=2)
ax.plot(x, np.exp(log_q(x)), linewidth=2)
leg = ax.legend(['True PDF', 'Initial', 'Optimized'], loc='best')
plt.setp(leg.get_texts(), fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
png_file = os.path.join('figures', 'optimize_1_elbo_results.png')
print 'writing:', png_file
plt.savefig(png_file)
