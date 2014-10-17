"""
A first test for the ELBO on the catalysis problem.

The target is consisted of an uninformative prior and a Gaussian likelihood.

The approximating mixture has two components.

Author:
    Panagiotis Tsilifis

Date:
    6/12/2014

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cPickle as pickle
from scipy.stats.distributions import norm
import math
from vuq import GammaPDF
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

from catalysis import CatalysisFullModelDMNLESS


# Number of dimensions
num_dim = 36 + 1

# The number of components to use for the mixture
num_comp = 1

#-------- The (hypothetical) joint distribution ----------------

# The prior
mu_prior = np.zeros(num_dim)
mu_prior[num_dim-1] = 0.1
C_prior = np.eye(num_dim) * 1.
C_prior[num_dim-1, num_dim-1] = 1e-1
prior = MultivariateNormal(mu_prior, C_prior)
# The data
data = np.loadtxt('data.txt').reshape((7, 6))
y = data[:, 1:]
y = y.reshape((1,y.shape[0] * y.shape[1]))
# The forward model
catal_model = CatalysisFullModelDMNLESS()
print 'Num_input'
print str(catal_model.num_input) + '\n'
# The isotropic Likelihood
IsotropicL = IsotropicGaussianLikelihood(y[0,:] / 500., catal_model)
# The joint
log_p = Joint(IsotropicL, prior)
print 'Target:'
print str(log_p)

# The approximating distribution
comp = [MultivariateNormal(np.random.gamma(10,1,num_dim))] #, MultivariateNormal(np.random.gamma(10,1,num_dim))]
log_q = MixtureOfMultivariateNormals(comp)
#log_q.comp[0].mu = np.ones(log_q.comp[0].mu.shape)
log_q.comp[0].mu = np.zeros(log_q.num_dim)
log_q.comp[0].mu[-1] = 1.
#log_q.comp[1].mu = log_q.comp[0].mu
log_q.comp[0].C = np.eye(num_dim) * 0.5
#log_q.comp[1].C = np.eye(num_dim) * 0.5
print 'Initial:'
print log_q

# Pick an entropy approximation
entropy = FirstOrderEntropyApproximation()
# Pick an approximation for the expectation of the joint
expectation_functional = ThirdOrderExpectationFunctional(log_p)
# Restrictions for mu
#mu_bounds = (tuple((-5, 8) for i in xrange(log_q.num_dim - 1))
mu_bounds = (tuple((-10, 10) for i in xrange(log_q.num_dim - 1))  + ((1e-6, None), ))

           
C_bounds = tuple((1e-32, None) for i in xrange(log_q.num_comp * log_q.num_dim))
# Make the constraints for A
def mu_c_fun(mu, j):
    n = int(math.sqrt(log_q.num_dim - 1))
    A = mu[:-1].reshape((n, n))
    return np.sum(A[:, j])

def mu_c_jac(mu, j):
    n = int(math.sqrt(log_q.num_dim - 1))
    A = mu[:-1].reshape((n, n))
    J = np.zeros((n, n))
    J[:, j] = 1.
    J_f = np.hstack([J.flatten(), 0.])
    return J_f
mu_constraints = []
n = int(math.sqrt(log_q.num_dim - 1))
for j in xrange(n):
    c = {}
    c['type'] = 'eq'
    c['fun'] = mu_c_fun
    c['jac'] = mu_c_jac
    c['args'] = (j, )
    mu_constraints.append(c)

# Build the ELBO
elbo = EvidenceLowerBound(entropy, expectation_functional)
print 'ELBO:'
print str(elbo)

# Optimize the elbo
optimizer = Optimizer(elbo)

results_file = os.path.join('demos', 'catalysis_full_dmnl_cali.pcl')
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
                           mu_constraints=mu_constraints,
                           C_bounds=C_bounds)
    result = {}
    result['L'] = L
    result['log_q'] = log_q
    with open(os.path.join('demos', 'catalysis_full_dmnl_cali.pcl'), 'wb') as fd:
        pickle.dump(result, fd)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(L, linewidth=2)
ax.set_xlabel('Iteration', fontsize=16)
ax.set_ylabel('ELBO', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
png_file = os.path.join('figures', 'catalysis_full_dmnl_elbo.png')
print 'Writing:', png_file
plt.savefig(png_file)

A = log_q.comp[0].mu[:-1].reshape(n, n)
print 'A = '
print A

for i in xrange(log_q.num_dim):
    mu = log_q.comp[0].mu[i]
    s = math.sqrt(log_q.comp[0].C[i, i])
    if i < 36:
        name = 'kappa_{%s}' % (i+1)
    else:
        name = 'sigma^2'
    print name, '=', mu, '+-', s

# Plot the calibration result
t = np.array([0.0, 1./6, 1./3, 1./2, 2./3, 5./6, 1.])
fig = plt.figure()
ax = fig.add_subplot(111)
m_state = catal_model(log_q.comp[0].mu[:36])
f = m_state['f']
Y = f.reshape(t.shape[0], f.shape[1] / t.shape[0])
styles = ['b', 'r', 'g', 'k', 'm']
for i in xrange(5):
    ax.plot(t, Y[:, i], styles[i], linewidth=2)
    ax.plot(t, data[:, 1:][:, i] / 500, '+' + styles[i], markersize=10, markeredgewidth=2)
ax.set_xlabel('Time (t)', fontsize=16)
ax.set_ylabel('Concentration', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
png_file = os.path.join('figures', 'catalysis_full_dmnl_cali_output.png')
print 'Writing:', png_file
plt.savefig(png_file)

# Do an uncertainty propagation test.
uq_file = os.path.join('demos', 'catalysis_full_dmnl_cali_uq.pcl')
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
        x = omega[:36]
        sigma = omega[36]
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
    ax.plot(t, data[:, 1:][:, i] / 500, '+' + styles[i], markersize=10,
            markeredgewidth=2)
ax.set_xlabel('Time (t)', fontsize=16)
ax.set_ylabel('Concentration', fontsize=16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
png_file = os.path.join('figures', 'catalysis_full_dmnl_cali_uq.png')
print 'Writing:', png_file
plt.savefig(png_file)
