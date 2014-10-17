"""
Test the entropy approximations.

Author:
    Ilias Bilionis

Date:
    6/3/2014
    9/17/2014

"""


import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from vuq import MixtureOfMultivariateNormals
from vuq import MonteCarloEntropyApproximation
from vuq import FirstOrderEntropyApproximation
from vuq import EntropyLowerBound


# Number of components for the test
num_dim = 1

# Create a random multivariate normal
q = MixtureOfMultivariateNormals.create(num_dim, 1) # With one component first

# Create the entropy approximation
S0 = FirstOrderEntropyApproximation()

# Create a monte carlo approximation to the entropy
Smc = MonteCarloEntropyApproximation(num_samples=100)

# The lower bound to the entropy
Sl = EntropyLowerBound()

# The first order approximation to the entropy
print 'S0[q] =', S0(q)['S']

# The lower bound to the entropy
print 'Sl[q] =', Sl.eval(q)

# Real entropy
print 'S[q] =', q.comp[0].entropy

# Plot it as a function of C
# Cs = np.linspace(0.1, 10, 100)
# s0 = []
# s = []
# sl = []
# for c in Cs:
#     q.comp[0].C = np.array([[c]])
#     s0.append(S0(q)['S'])
#     sl.append(Sl.eval(q))
#     s.append(q.comp[0].entropy)
# plt.plot(s)
# plt.plot(s0)
# plt.plot(sl)
# plt.legend(['S0', 'Sl', 'S'])
# plt.show()

# Try it out with more components
print 'Doing it with two components...'
q = MixtureOfMultivariateNormals.create(num_dim, 2)
print str(q)
print 'S0[q] =', S0(q)['S']
print 'Smc[q] =', Smc(q)
print 'Sl[q] = ', Sl.eval(q)

# Now, let's vary one of the mu's and see what happens
mus = np.linspace(-10., 10., 100)
s0 = []
smc = []
sl = []
for mu in mus:
    q.comp[0].mu = np.array([mu])
    s0.append(S0(q)['S'])
    sl.append(Sl.eval(q))
    smc.append(Smc(q))
plt.plot(mus, smc, 'g', linewidth=2)
plt.plot(mus, s0, 'b--', linewidth=2)
plt.plot(mus, sl, 'r:', linewidth=2)
plt.plot(q.comp[1].mu[0], np.min(s0), 'or', markersize=10)
leg = plt.legend(['$S[q]$ (via MC)', '$S_1[q]$', '$S_l[q]$', '$\mu_2$'], loc='best')
plt.setp(leg.get_texts(), fontsize=16)
plt.xlabel('$\mu_1$', fontsize=16)
plt.ylabel('Entropy', fontsize=16)
plt.title(r'Entropy of $q(x) = \frac{1}{2}\mathcal{N}(x|\mu_1, 1) + \frac{1}{2}\mathcal{N}(x|\mu_2, 1)$', fontsize=16)
plt.setp(plt.axes().get_xticklabels(), fontsize=16)
plt.setp(plt.axes().get_yticklabels(), fontsize=16)
plt.show()

# Let's see what happens to the same plot if we double the covariance
q.comp[0].C = np.array([[2.]])
s0_d = []
sl_d = []
for mu in mus:
    q.comp[0].mu = np.array([mu])
    s0_d.append(S0(q)['S'])
    sl_d.append(Sl.eval(q))
plt.plot(mus, s0, 'b--', linewidth=2)
plt.plot(mus, s0_d, 'g', linewidth=2)
plt.plot(mus, sl_d, 'r:', linewidth=2)
plt.plot(q.comp[1].mu[0], np.min(s0), 'or', markersize=10)
leg = plt.legend(['$S_1[q], C=1$', '$S_1[q], C=2$', '$S_l[q]$', '$\mu_2$'], loc='best')
plt.setp(leg.get_texts(), fontsize=16)
plt.xlabel('$\mu_1$', fontsize=16)
plt.ylabel('Entropy', fontsize=16)
plt.title(r'Entropy of $q(x) = \frac{1}{2}\mathcal{N}(x|\mu_1, C) + \frac{1}{2}\mathcal{N}(x|\mu_2, 1)$', fontsize=16)
plt.setp(plt.axes().get_xticklabels(), fontsize=16)
plt.setp(plt.axes().get_yticklabels(), fontsize=16)
plt.show()

