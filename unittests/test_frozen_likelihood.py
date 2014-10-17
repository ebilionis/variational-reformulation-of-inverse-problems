"""
Test how the likelihood functions work.

Author:
    Panagiotis Tsilifis
    Ilias Bilionis

Date:
    5/26/2014
    6/6/2014
"""

import numpy as np
import time
from vuq import IndepNoiseGaussianLikelihood
from vuq import IsotropicGaussianLikelihood

import sys
sys.path.insert(0,'demos/')

from catalysis import CatalysisModel

# Define the data first
data = np.loadtxt('data.txt').reshape((7, 6))
y = data[:, 1:]
y = y.reshape((1,y.shape[0] * y.shape[1]))
# Define the model
kappa = 0.1*np.random.rand(5)
catal_model = CatalysisModel(kappa)

# Define the isotropic Likelihood
like = IsotropicGaussianLikelihood(y[0,:], catal_model)

fixed_theta_like = like.freeze([10.])

print 'The frozen likelihood:'
print str(fixed_theta_like)

# Evaluate it at some kappas
kappa = 0.1 * np.random.rand(10, 5)
t0 = time.time()
L, dLdx, d2Ldx2 = fixed_theta_like(kappa)
t1 = time.time()
print 'L'
print L
print 'dLdx'
print dLdx
#print 'Elapsed time (cache not active):', t1 - t0
#print 'Let us do it again on the same points.'
#print 'Now the cache of the forward model should be active.'
#t0 = time.time()
#L, dLdx, d2Ldx2 = fixed_theta_like(kappa)
#t1 = time.time()
#print 'Elapsed time (cache is active):', t1 - t0
