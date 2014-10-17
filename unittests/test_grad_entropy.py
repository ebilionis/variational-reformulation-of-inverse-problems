"""
Test the gradient of entropy approximations.

Author:
    Ilias Bilionis

Date:
    6/3/2014

"""


import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from vuq import MixtureOfMultivariateNormals
from vuq import MonteCarloEntropyApproximation
from vuq import FirstOrderEntropyApproximation
from vuq import EntropyLowerBound


def dn_mu(S0, q, h=1e-6):
    """
    Get numerical derivatives with respect to mu.
    """
    S = S0.eval(q)[0]
    mu = np.copy(q.mu)
    mu_new = np.zeros(mu.shape)
    J = np.zeros((mu.shape[0], 1, mu.shape[1]))
    for i in xrange(mu.shape[0]):
        for j in xrange(mu.shape[1]):
            mu_new[:] = mu
            mu_new[i, j] += h
            q.mu = mu_new
            Sph = S0.eval(q)[0]
            J[i, 0, j] = (Sph - S) / h
            q.mu = mu
    return J


def dn_C(S0, q, h=1e-6):
    """
    Get numerical derivatives with respect to C.
    """
    S = S0.eval(q)[0]
    C = np.copy(q.C)
    C_new = np.zeros(C.shape)
    J = np.zeros(C.shape)
    for i in xrange(C.shape[0]):
        for j in xrange(C.shape[1]):
            for k in xrange(C.shape[2]):
                C_new[:] = C
                C_new[i, j, k] += h
                if k != j:
                    C_new[i, k, j] += h
                q.C = C_new
                Sph = S0.eval(q)[0]
                J[i, j, k] = (Sph - S) / h
                if k != j:
                    J[i, j, k] *= 0.5
                q.C = C
    return J


# Number of components for the test
num_dim = 1

# Create a random multivariate normal
q = MixtureOfMultivariateNormals.create(num_dim, 2) # With one component first

# Create the entropy approximation
#S0 = FirstOrderEntropyApproximation()
S0 = EntropyLowerBound()

# Create a monte carlo approximation to the entropy
Smc = MonteCarloEntropyApproximation(num_samples=100)

# Try it out with more components
print 'Doing it with two components...'
q = MixtureOfMultivariateNormals.create(num_dim, 2)
#q.comp[0].C = np.array([[2]])
#q.w = np.array([0.3, 0.7])
print str(q)
# Evaluate the entropy
s0, grad_w, grad_mu, grad_C = S0(q)
state = S0(q)
s0 = state['S']
grad_w = state['S_grad_w']
grad_mu = state['S_grad_mu']
grad_C = state['S_grad_C']
# Check the numerical derivatives
n_grad_mu = dn_mu(S0, q)
n_grad_C = dn_C(S0, q)
print 'S0[q] =', s0
print 'grad w:'
print grad_w
print 'grad mu:'
print grad_mu.shape
print grad_mu
print 'n. grad mu:'
print n_grad_mu
print 'error in grad mu:'
print grad_mu - n_grad_mu
print 'grad C:'
print grad_C.shape
print grad_C
print 'n. grad C:'
print n_grad_C
print 'error in grad_C:'
print grad_C - n_grad_C

# What follows will work only if num_dim == 1
if num_dim != 1:
    quit()

# Now, let's vary one of the mu's and see what happens
mus = np.linspace(-10., 10., 100)
s0 = []
s_grad_mu = []
#smc = []
for mu in mus:
    q.comp[0].mu = np.array([mu])
    state = S0(q)
    s00 = state['S']
    grad_mu = state['S_grad_mu']
    s0.append(s00)
    s_grad_mu.append(grad_mu[0, 0, 0])
#    smc.append(Smc(q))
#plt.plot(mus, smc, 'g', linewidth=2)

fig, ax1 = plt.subplots()
ax1.plot(mus, s0, 'b-', linewidth=2)
ax1.set_xlabel('$\mu_1$', fontsize=16)
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('$S_1[q]$', color='b', fontsize=16)
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
ax2.plot(mus, s_grad_mu, 'r', linewidth=2)
ax2.set_ylabel(r'$\nabla_{\mu_1}S_1[q]$', color='r', fontsize=16)
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.show()
