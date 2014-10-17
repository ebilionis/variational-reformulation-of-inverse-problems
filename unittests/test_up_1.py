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
from vuq import UncertaintyPropagationLikelihood
from vuq import FlatPDF
from vuq import MultivariateNormal
from vuq import PDFCollection
from vuq import Joint
from vuq import MixturePDF
from vuq import MixtureOfMultivariateNormals
from vuq import FirstOrderEntropyApproximation
from vuq import ThirdOrderExpectationFunctional
from vuq import EvidenceLowerBound
from vuq import FullOptimizer
from demos import TestModel1

# The number of components to use for the mixture
num_comp = 5

# The model
model = TestModel1()

# The prior
log_p_x = MultivariateNormal(mu=[0])
log_p_z_fake = FlatPDF(model.num_output)
log_p_x_ext = PDFCollection([log_p_x, log_p_z_fake])
# The isotropic Likelihood
log_p_z_given_x = UncertaintyPropagationLikelihood(model, alpha=1e-1)
# The joint
log_p = Joint(log_p_z_given_x, log_p_x_ext)

# The approximating distribution
log_q = MixtureOfMultivariateNormals.create(log_p.num_dim, num_comp)

# Build the ELBO
# Pick an entropy approximation
entropy = FirstOrderEntropyApproximation()
# Pick an approximation for the expectation of the joint
expectation_functional = ThirdOrderExpectationFunctional(log_p)
# Build the ELBO
elbo = EvidenceLowerBound(entropy, expectation_functional)
print 'ELBO:'
print str(elbo)

# Optimize the elbo
optimizer = FullOptimizer(elbo)

C_bounds = tuple((1e-32, None) for i in xrange(log_q.num_comp * log_q.num_dim))
L = optimizer.optimize(log_q, max_it=10)#, C_bounds=C_bounds)

print 'Result:'
print log_q

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(L, linewidth=2)
ax.set_xlabel('Iteration', fontsize=16)
ax.set_ylabel('ELBO', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
png_file = os.path.join('figures', 'test_up_1_elbo.png')
print 'Writing:', png_file
plt.savefig(png_file)
quit()

for i in xrange(log_q.num_dim):
    mu = log_q.comp[0].mu[i]
    s = math.sqrt(log_q.comp[0].C[i, i])
    if i < 5:
        name = 'kappa_{%s}' % (i+1)
    else:
        name = 'sigma^2'
    print name, '=', mu, '+-', s

# Plot the calibration result
t = np.array([0.0, 30., 60., 90., 120., 150., 180.])
fig = plt.figure()
ax = fig.add_subplot(111)
m_state = catal_model(log_q.comp[0].mu[:5])
f = m_state['f']
Y = f.reshape(t.shape[0], f.shape[1] / t.shape[0])
styles = ['b', 'r', 'g', 'k', 'm']
for i in xrange(5):
    ax.plot(t, Y[:, i], styles[i], linewidth=2)
    ax.plot(t, data[:, 1:][:, i], '+' + styles[i], markersize=10, markeredgewidth=2)
ax.set_xlabel('Time (t)', fontsize=16)
ax.set_ylabel('Concentration', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
png_file = os.path.join('figures', 'catalysis_1_cali_output.png')
print 'Writing:', png_file
plt.savefig(png_file)

# Do an uncertainty propagation test.
uq_file = os.path.join('demos', 'catalysis_1_cali_uq.pcl')
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
for i in xrange(5):
    ax.plot(t, Y_m[:, i], styles[i], linewidth=2)
    ax.fill_between(t, Y_p05[:, i], Y_p95[:, i], color=styles[i], alpha=0.5)
    ax.plot(t, data[:, 1:][:, i], '+' + styles[i], markersize=10,
            markeredgewidth=2)
ax.set_xlabel('Time (t)', fontsize=16)
ax.set_ylabel('Concentration', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
png_file = os.path.join('figures', 'catalysis_1_cali_uq.png')
print 'Writing:', png_file
plt.savefig(png_file)
