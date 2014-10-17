"""
A first test for the ELBO on the catalysis problem.

The target is consisted of an uninformative prior and a Gaussian likelihood.

The approximating mixture has two components.

Author:
    Panagiotis Tsilifis

Date:
    6/6/2014

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cPickle as pickle
from scipy.stats.distributions import norm
import math
from vuq import GammaPDF
from vuq import PDFCollection
from vuq import UninformativePDF
from vuq import ProportionalNoiseLikelihood
from vuq import LikelihoodCollection
from vuq import MultivariateNormal
from vuq import Joint
from vuq import MixturePDF
from vuq import MixtureOfMultivariateNormals
from vuq import FirstOrderEntropyApproximation
from vuq import ThirdOrderExpectationFunctional
from vuq import EvidenceLowerBound
from vuq import Optimizer
import sys
sys.path.insert(0,'demos/')

from catalysis import CatalysisModelDMNLY0 as CatalysisModel


# Number of dimensions
num_dim = 6

# The number of components to use for the mixture
num_comp = 1

#-------- The (hypothetical) joint distribution ----------------

# The prior
#collection = [GammaPDF(10, 1, 1) for i in xrange(num_dim-1) ]
#collection = np.hstack([collection, GammaPDF(5,0.1,1)])
#prior = PDFCollection(collection)
prior = UninformativePDF(num_dim)
# The data
data = np.loadtxt('data.txt').reshape((7, 6))
y = data[1:, 1:] / 500.
y = y.reshape((y.shape[0] * y.shape[1], ))
# The forward model
catal_model = CatalysisModel()
print 'Num_input'
print str(catal_model.num_input) + '\n'
# The isotropic Likelihood
like = ProportionalNoiseLikelihood(y, catal_model)
# The joint
log_p = Joint(like, prior)
print 'Target:'
print str(log_p)

# The approximating distribution
comp = [MultivariateNormal(np.random.gamma(10,1,num_dim))]#, MultivariateNormal(np.random.gamma(10,1,num_dim))]
log_q = MixtureOfMultivariateNormals(comp)
mu = np.random.rand(num_dim)
log_q.comp[0].mu = mu
log_q.comp[0].C = np.eye(num_dim) * 0.5
print 'Initial:'
print log_q

# Pick an entropy approximation
entropy = FirstOrderEntropyApproximation()
# Pick an approximation for the expectation of the joint
expectation_functional = ThirdOrderExpectationFunctional(log_p)
# Restrictions for mu
mu_bounds = (tuple((1e-6, None) for i in xrange(log_q.num_dim - 1))
            + ((1e-6, None), ))
C_bounds = tuple((1e-10, None) for i in xrange(log_q.num_comp * log_q.num_dim))
# Build the ELBO
elbo = EvidenceLowerBound(entropy, expectation_functional)
print 'ELBO:'
print str(elbo)

# Optimize the elbo
optimizer = Optimizer(elbo)

results_file = os.path.join('demos', 'catalysis_prop_noise_cali.pcl')
if os.path.exists(results_file):
    print 'I found:', results_file
    print 'I am skipping the experiment.'
    print 'Delete the file if you want to repeat it.'
    with open(results_file, 'rb') as fd:
        results = pickle.load(fd)
    L = results['L']
    log_q = results['log_q']
else:
    L = optimizer.optimize(log_q, max_it=10, mu_bounds=mu_bounds,
                           C_bounds=C_bounds)
    result = {}
    result['L'] = L
    result['log_q'] = log_q
    with open(os.path.join('demos', 'catalysis_prop_noise_cali.pcl'), 'wb') as fd:
        pickle.dump(result, fd)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(L, linewidth=2)
ax.set_xlabel('Iteration', fontsize=16)
ax.set_ylabel('ELBO', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
png_file = os.path.join('figures', 'catalysis_prop_noise_elbo.png')
print 'Writing:', png_file
plt.savefig(png_file)

for i in xrange(log_q.num_dim):
    mu = log_q.comp[0].mu[i]
    s = math.sqrt(log_q.comp[0].C[i, i])
    if i < 5:
        name = 'kappa_{%s}' % (i+1)
    else:
        name = 'sigma^2'
    print name, '=', mu, '+-', s

# Plot the calibration result
t = np.array([30., 60., 90., 120., 150., 180.]) / 180.
fig = plt.figure()
ax = fig.add_subplot(111)
m_state = catal_model(log_q.comp[0].mu[:5])
f = m_state['f']
Y = f.reshape(t.shape[0], f.shape[1] / t.shape[0])
styles = ['b', 'r', 'g', 'k', 'm']
for i in xrange(5):
    ax.plot(t, Y[:, i], styles[i], linewidth=2)
    ax.plot(t, data[1:, 1:][:, i] / 500., '+' + styles[i], markersize=10, markeredgewidth=2)
ax.set_xlabel('Time (t)', fontsize=16)
ax.set_ylabel('Concentration', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
png_file = os.path.join('figures', 'catalysis_prop_noise_cali_output.png')
print 'Writing:', png_file
plt.savefig(png_file)

# Do an uncertainty propagation test.
uq_file = os.path.join('demos', 'catalysis_prop_noise_cali_uq.pcl')
if os.path.exists(uq_file):
    with open(uq_file, 'rb') as fd:
        uq_results = pickle.load(fd)
    Y_m = uq_results['Y_m']
    Y_p05 = uq_results['Y_p05']
    Y_p95 = uq_results['Y_p95']
else:
    num_mcmc = 100
    Y_s = []
    for i in xrange(num_mcmc):
        print 'taking sample', i + 1
        omega = log_q.sample().flatten()
        x = omega[:5]
        sigma = omega[5]
        y = catal_model(x)['f']
        Y_s.append(y + sigma * y * np.random.randn(*y.shape))
    Y_s = np.vstack(Y_s)
    Y_m = np.percentile(Y_s, 50, axis=0).reshape(Y.shape)
    Y_p05 = np.percentile(Y_s, 5, axis=0).reshape(Y.shape)
    Y_p95 = np.percentile(Y_s, 95, axis=0).reshape(Y.shape)
    uq_results = {}
    uq_results['Y_m'] = Y_m
    uq_results['Y_p05'] = Y_p05
    uq_results['Y_p95'] = Y_p95
    with open(uq_file, 'wb') as fd:
        pickle.dump(uq_results, fd)

fig = plt.figure()
ax = fig.add_subplot(111)
for i in xrange(5):
    ax.plot(t, Y_m[:, i], styles[i], linewidth=2)
    ax.fill_between(t, Y_p05[:, i], Y_p95[:, i], color=styles[i], alpha=0.5)
    ax.plot(t, data[1:, 1:][:, i] / 500., '+' + styles[i], markersize=10,
            markeredgewidth=2)
ax.set_xlabel('Time (t)', fontsize=16)
ax.set_ylabel('Concentration', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
png_file = os.path.join('figures', 'catalysis_prop_noise_cali_uq.png')
print 'Writing:', png_file
plt.savefig(png_file)
