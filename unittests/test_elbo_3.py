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
from vuq import MultivariateNormal
from vuq import MixtureOfMultivariateNormals
from vuq import FirstOrderEntropyApproximation
from vuq import ThirdOrderExpectationFunctional
from vuq import EvidenceLowerBound


# Number of dimensions
num_dim = 1

# The number of components to use for the mixture
num_comp = 2

# The (hypothetical) joint distribution
log_p = MixtureOfMultivariateNormals.create(num_dim, 2)
log_p.comp[0].mu = [-1.]
log_p.comp[0].C = [[0.2]]

log_p.comp[1].mu = [1.]
log_p.comp[1].C = [[0.2]]
print 'Target:'
print str(log_p)

# The approximating distribution
log_q = MixtureOfMultivariateNormals.create(num_dim, num_comp)
print 'Initial:'
print log_q
log_q.comp[0].C = [[0.8]]
log_q.comp[1].mu = [1.]
log_q.comp[1].C = [[0.8]]

# Pick an entropy approximation
entropy = FirstOrderEntropyApproximation()
# Pick an approximation for the expectation of the joint
expectation_functional = ThirdOrderExpectationFunctional(log_p)
# Build the ELBO
elbo = EvidenceLowerBound(entropy, expectation_functional)
print 'ELBO:'
print str(elbo)

# Evaluate the elbo
state = elbo(log_q)
print state

# Plot the elbo as a function of mu
mus = np.linspace(-2, 2, 100)
L = []
S = []
F = []
for mu in mus:
    log_q.comp[0].mu = [mu]
    state = elbo(log_q)
    L.append(state['L'])
    S.append(state['S_state']['S'])
    F.append(state['F_state']['F'])
L = np.hstack(L)
S = np.hstack(S)
F = np.hstack(F)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(mus, L, linewidth=2)
ax.plot(mus, S, linewidth=2)
ax.plot(mus, F, linewidth=2)
leg = ax.legend(['ELBO', 'Entropy', 'Expectation'], loc='best')
plt.setp(leg.get_texts(), fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
ax.set_xlabel('$\mu$', fontsize=16)
png_file = os.path.join('figures', 'test_elbo_3_varying_mu.png')
print 'writing:', png_file
plt.savefig(png_file)

# Now fix mu and vary C
log_q.comp[0].mu = [-1.]
Cs = np.linspace(0.01, 1.2, 100)
L = []
S = []
F = []
for C in Cs:
    log_q.comp[0].C = [[C]]
    state = elbo(log_q)
    L.append(state['L'])
    S.append(state['S_state']['S'])
    F.append(state['F_state']['F'])
L = np.hstack(L)
S = np.hstack(S)
F = np.hstack(F)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Cs, L, linewidth=2)
ax.plot(Cs, S, linewidth=2)
ax.plot(Cs, F, linewidth=2)
leg = ax.legend(['ELBO', 'Entropy', 'Expectation'], loc='best')
plt.setp(leg.get_texts(), fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
ax.set_xlabel('$C$', fontsize=16)
png_file = os.path.join('figures', 'test_elbo_3_varying_C.png')
print 'writing:', png_file
plt.savefig(png_file)

# Now do a contour plot
mus = np.linspace(-1.3, 1.3, 64)
Cs = np.linspace(0.01, 1.2, 64)
Mus, CCs = np.meshgrid(mus, Cs)
Z = np.zeros(Mus.shape)
for i in xrange(Z.shape[0]):
    for j in xrange(Z.shape[1]):
        log_q.comp[0].mu = [Mus[i, j]]
        log_q.comp[0].C = [[CCs[i, j]]]
        state = elbo(log_q)
        Z[i, j] = state['L']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.contourf(Mus, CCs, Z)
ax.plot(log_p.mu.flatten(), log_p.C.flatten(), 'ok', markersize=16)
ax.set_xlabel('$\mu$', fontsize=16)
ax.set_ylabel('$C$', fontsize=16)
ax.set_title('ELBO as a function of $\mu$ and $C$', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
cbar = fig.colorbar(cax)
plt.setp(cbar.ax.get_xticklabels(), fontsize=16)
png_file = os.path.join('figures', 'test_elbo_3_varying_both.png')
print 'writing:', png_file
plt.savefig(png_file)

# Now fix the Cs and vary both mus
log_q.comp[0].C = log_p.comp[0].C
mus = np.linspace(-4, 4, 64)
Mus1, Mus2 = np.meshgrid(mus, mus)
Z = np.zeros(Mus.shape)
for i in xrange(Z.shape[0]):
    for j in xrange(Z.shape[1]):
        log_q.comp[0].mu = [Mus1[i, j]]
        log_q.comp[1].mu = [Mus2[i, j]]
        state = elbo(log_q)
        Z[i, j] = state['L']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.contourf(Mus1, Mus2, Z)
ax.plot(log_p.mu[0, 0], log_p.mu[1, 0], 'ok', markersize=16)
ax.set_xlabel('$\mu_1$', fontsize=16)
ax.set_ylabel('$\mu_2$', fontsize=16)
ax.set_title('ELBO as a function of $\mu_1$ and $\mu_2$', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
cbar = fig.colorbar(cax)
plt.setp(cbar.ax.get_xticklabels(), fontsize=16)
png_file = os.path.join('figures', 'test_elbo_3_varying_both_mus.png')
print 'writing:', png_file
plt.savefig(png_file)
