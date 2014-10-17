"""
A first test for the ELBO on the diffusion problem.

The target is consisted of an and a Gaussian likelihood.

The approximating mixture has two components.

Author:
    Panagiotis Tsilifis

Date:
    6/16/2014

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cPickle as pickle
from scipy.stats.distributions import norm
import math
from vuq import GammaPDF
from vuq import UniformND
from vuq import PDFCollection
from vuq import IsotropicGaussianLikelihood
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

from diffusion import ContaminantTransportModelCenter

# Number of dimensions
num_dim = 3

# The number of components to use for the mixture
num_comp = 2

#-------- The (hypothetical) joint distribution ----------------

# The prior
collection = [UniformND(1), UniformND(1), GammaPDF(1,0.05,1)]
prior = PDFCollection(collection)
# The data
data = np.load('data_concentrations_upperlowercenters.npy')
# The forward model
diff_model = ContaminantTransportModelCenter()
print 'Num_input'
print str(diff_model.num_input) + '\n'
# The isotropic Likelihood
IsotropicL = IsotropicGaussianLikelihood(data[:], diff_model)
# The joint
log_p = Joint(IsotropicL, prior)
print 'Target:'
print str(log_p)

# The approximating distribution
comp = [MultivariateNormal(np.random.gamma(10,1,num_dim)), MultivariateNormal(np.random.gamma(10,1,num_dim))]
log_q = MixtureOfMultivariateNormals(comp)
log_q.comp[0].mu[0] = 0.1
log_q.comp[0].mu[1:] = np.ones(2) * 0.5
log_q.comp[1].mu[0] = 0.9
log_q.comp[1].mu[1:] = np.ones(2) * 0.5
log_q.comp[0].C = np.eye(num_dim) * 1e-4
log_q.comp[1].C = np.eye(num_dim) * 1e-4
print 'Initial:'
print log_q

# Pick an entropy approximation
entropy = FirstOrderEntropyApproximation()
# Pick an approximation for the expectation of the joint
expectation_functional = ThirdOrderExpectationFunctional(log_p)
# Restrictions for mu
mu_bounds = (tuple((0., 1.) for i in xrange(log_q.num_dim - 1))
            + ((1e-6, None), ))
C_bounds = tuple((1e-32, None) for i in xrange(log_q.num_comp * log_q.num_dim))
# Build the ELBO
elbo = EvidenceLowerBound(entropy, expectation_functional)
print 'ELBO:'
print str(elbo)


# Optimize the elbo
optimizer = Optimizer(elbo)

results_file = os.path.join('demos', 'diffusion_centers1_cali.pcl')
if os.path.exists(results_file):
    print 'I found:', results_file
    print 'I am skipping the experiment.'
    print 'Delete the file if you want to repeat it.'
    with open(results_file, 'rb') as fd:
        results = pickle.load(fd)
    L = results['L']
    log_q = results['log_q']
else:
    L = optimizer.optimize(log_q, tol=1e-4, max_it=10, mu_bounds=mu_bounds,
                           mu_constraints=None, C_bounds=C_bounds)
    result = {}
    result['L'] = L
    result['log_q'] = log_q
    with open(os.path.join('demos', 'diffusion_centers1_cali.pcl'), 'wb') as fd:
        pickle.dump(result, fd)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(L, linewidth=2)
ax.set_xlabel('Iteration', fontsize=16)
ax.set_ylabel('ELBO', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
png_file = os.path.join('figures', 'diffusion_centers1_elbo.png')
print 'Writing:', png_file
plt.savefig(png_file)

for i in xrange(log_q.num_dim):
    mu = log_q.comp[0].mu[i]
    s = math.sqrt(log_q.comp[0].C[i, i])
    if i < 2:
        name = 'x_{%s}' % (i+1)
    else:
        name = 'sigma^2'
    print name, '=', mu, '+-', s

# Plot the calibration result
t = np.array([ 0.075, 0.15, 0.225, 0.3])
fig = plt.figure()
ax = fig.add_subplot(111)
f = diff_model._eval_u(log_q.comp[0].mu[:2])
Y = f.reshape(4, 2)
data = data.reshape(4, 2)
styles = ['b', 'r']
for i in xrange(2):
    ax.plot(t, Y[:, i], styles[i], linewidth=2)
    ax.plot(t, data[:,i], '+' + styles[i], markersize=10, markeredgewidth=2)
ax.set_xlabel('Time (t)', fontsize=16)
ax.set_ylabel('Concentration', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
png_file = os.path.join('figures', 'diffusion_centers1_cali_output.png')
print 'Writing:', png_file
plt.savefig(png_file)

# Do an uncertainty propagation test.
uq_file = os.path.join('demos', 'diffusion_centers1_cali_uq.pcl')
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
        x = omega[:2]
        sigma = omega[2]
        y = diff_model._eval_u(x)
        Y_s.append(y + sigma * np.random.randn(*y.shape))
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
for i in xrange(2):
    ax.plot(t, Y_m[:, i], styles[i], linewidth=2)
    ax.fill_between(t, Y_p05[:, i], Y_p95[:, i], color=styles[i], alpha=0.5)
    ax.plot(t, data[:, i], '+' + styles[i], markersize=10,
            markeredgewidth=2)
ax.set_xlabel('Time (t)', fontsize=16)
ax.set_ylabel('Concentration', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
png_file = os.path.join('figures', 'diffusion_centers1_cali_uq.png')
print 'Writing:', png_file
plt.savefig(png_file)

print str(log_q) + '\n'
