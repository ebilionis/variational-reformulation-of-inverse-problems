"""
A very simple test for the ELBO.

The target is now a mixture of two Gaussians.

The approximating mixture has two components.

Author:
    Ilias Bilionis

Date:
    6/5/2014

"""


import numpy as np
import matplotlib.pyplot as plt
import os
from vuq import *


# Number of dimensions
num_dim = 2

# The number of components to use for the mixture
num_comp = 10

# The (hypothetical) joint distribution
comp = [MultivariateT([-1, -1], 4), MultivariateT([1, 1], 4)]
log_p = MixturePDF(comp)
print 'Target:'
print str(log_p)

# The approximating distribution
log_q = MixtureOfMultivariateNormals.create(num_dim, num_comp)
for i in xrange(num_comp):
    log_q.comp[i].C = np.eye(num_dim) * 0.5
print 'Initial:'
print log_q

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
png_file = os.path.join('figures', 'optimize_2_elbo.png')
print 'writing:', png_file
plt.savefig(png_file)

# Let's plot the distributions
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
X1, X2 = np.meshgrid(x, x)
XX = np.hstack([X1.flatten()[:, None], X2.flatten()[:, None]])
Z_true = log_p(XX).reshape(X1.shape)
cax1 = ax1.contourf(X1, X2, np.exp(Z_true))
cbar1 = fig1.colorbar(cax1)
png_file = os.path.join('figures', 'optimize_2_elbo_true.png')
print 'writing:', png_file
plt.savefig(png_file)
plt.clf()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
Z = log_q(XX).reshape(X1.shape)
cax2 = ax2.contourf(X1, X2, np.exp(Z))
cbar2 = fig2.colorbar(cax2)
png_file = os.path.join('figures', 'optimize_2_elbo_results.png')
print 'writing:', png_file
plt.savefig(png_file)
#leg = ax.legend(['True PDF', 'Initial', 'Optimized'], loc='best')
#plt.setp(leg.get_texts(), fontsize=16)
#plt.setp(ax.get_xticklabels(), fontsize=16)
#plt.setp(ax.get_yticklabels(), fontsize=16)

