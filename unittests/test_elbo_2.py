"""
A very simple test for the ELBO.

The target is now a mixture of two Gaussians.

The approximating mixture has one component.

Author:
    Ilias Bilionis

Date:
    6/5/2014
    9/17/2014

"""


import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from vuq import MultivariateNormal
from vuq import MixtureOfMultivariateNormals
from vuq import FirstOrderEntropyApproximation
from vuq import EntropyLowerBound
from vuq import ThirdOrderExpectationFunctional
from vuq import EvidenceLowerBound


# Number of dimensions
num_dim = 1

# The number of components to use for the mixture
num_comp = 1

# The (hypothetical) joint distribution
log_p = MixtureOfMultivariateNormals.create(num_dim, 2)
log_p.comp[0].mu = [-1.]
log_p.comp[0].C = [[0.5]]

log_p.comp[1].mu = [1.]
log_p.comp[1].C = [[0.5]]
print 'Target:'
print str(log_p)

# The approximating distribution
log_q = MixtureOfMultivariateNormals.create(num_dim, num_comp)
print 'Initial:'
print log_q

# Pick an entropy approximation
entropy = FirstOrderEntropyApproximation()
entropy_lb = EntropyLowerBound()
# Pick an approximation for the expectation of the joint
expectation_functional = ThirdOrderExpectationFunctional(log_p)
# Build the ELBO
elbo = EvidenceLowerBound(entropy, expectation_functional)
elbo_2 = EvidenceLowerBound(entropy_lb, expectation_functional)
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
L2 = []
S2 = []
for mu in mus:
    log_q.comp[0].mu = [mu]
    state = elbo(log_q)
    L.append(state['L'])
    S.append(state['S_state']['S'])
    F.append(state['F_state']['F'])
    state_2 = elbo_2(log_q)
    L2.append(state_2['L'])
    S2.append(state_2['S_state']['S'])
L = np.hstack(L)
S = np.hstack(S)
F = np.hstack(F)
L2 = np.hstack(L2)
S2 = np.hstack(S2)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(mus, L2, linewidth=2)
ax.plot(mus, S2, linewidth=2)
ax.plot(mus, F, linewidth=2)
#ax.plot(mus, L2, linewidth=2)
#ax.plot(mus, S2, linewidth=2)
leg = ax.legend(['ELBO', 'Entropy', 'Expectation'], loc='best')
plt.setp(leg.get_texts(), fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
ax.set_xlabel('$\mu$', fontsize=16)
png_file = os.path.join('figures', 'test_elbo_2_varying_mu.png')
print 'writing:', png_file
plt.savefig(png_file)

# Now fix mu and vary C
log_q.comp[0].mu = [0]
Cs = np.linspace(0.01, 2., 100)
L = []
S = []
F = []
L2 = []
S2 = []
for C in Cs:
    log_q.comp[0].C = [[C]]
    state = elbo(log_q)
    L.append(state['L'])
    S.append(state['S_state']['S'])
    F.append(state['F_state']['F'])
    state_2 = elbo_2(log_q)
    L2.append(state_2['L'])
    S2.append(state_2['S_state']['S'])
L = np.hstack(L)
S = np.hstack(S)
F = np.hstack(F)
L2 = np.hstack(L2)
S2 = np.hstack(S2)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Cs, L2, linewidth=2)
ax.plot(Cs, S2, linewidth=2)
ax.plot(Cs, F, linewidth=2)
#ax.plot(Cs, L2, linewidth=2)
#ax.plot(Cs, S2, linewidth=2)
leg = ax.legend(['ELBO', 'Entropy', 'Expectation'], loc='best')
plt.setp(leg.get_texts(), fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
ax.set_xlabel('$C$', fontsize=16)
png_file = os.path.join('figures', 'test_elbo_2_varying_C.png')
print 'writing:', png_file
plt.savefig(png_file)

# Now do a contour plot
mus = np.linspace(-1.3, 1.3, 64)
Cs = np.linspace(0.01, 1.5, 64)
Mus, CCs = np.meshgrid(mus, Cs)
Z = np.zeros(Mus.shape)
for i in xrange(Z.shape[0]):
    for j in xrange(Z.shape[1]):
        log_q.comp[0].mu = [Mus[i, j]]
        log_q.comp[0].C = [[CCs[i, j]]]
        state = elbo_2(log_q)
        Z[i, j] = state['L']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.contourf(Mus, CCs, Z, levels=np.linspace(-6.4, 0., 20))
ax.plot(log_p.mu.flatten(), log_p.C.flatten(), 'ok', markersize=16)
ax.set_xlabel('$\mu$', fontsize=16)
ax.set_ylabel('$C$', fontsize=16)
ax.set_title('ELBO as a function of $\mu$ and $C$', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
cbar = fig.colorbar(cax)
plt.setp(cbar.ax.get_xticklabels(), fontsize=16)
png_file = os.path.join('figures', 'test_elbo_2_varying_both.png')
print 'writing:', png_file
plt.savefig(png_file)
