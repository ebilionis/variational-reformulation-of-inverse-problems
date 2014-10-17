"""
Test how the joint work. 

Author:
    Panagiotis Tsilifis

Date:
    5/30/2014

"""

import numpy as np
from vuq import Joint
from vuq import IsotropicGaussianLikelihood
from vuq import UninformativePDF
import sys
sys.path.insert(0,'demos/')

from catalysis import CatalysisModel


# The number of parameters
num_input = 6

# The prior pdf
log_p = UninformativePDF(num_input)

# Define the data 
data = np.loadtxt('data.txt').reshape((7, 6))
y = data[:, 1:]
y = y.reshape((1,y.shape[0] * y.shape[1]))
# Define the model
kappa = 0.1*np.random.rand(5)
catal_model = CatalysisModel()

# Define the isotropic Likelihood
IsotropicL = IsotropicGaussianLikelihood(y[0,:], catal_model)

# Define the log-joint pdf
log_f = Joint(IsotropicL, log_p)

print str(log_f) + '\n'

# Evaluate Joint at x 
x = 0.1 * np.random.rand(num_input - 1)
sig = np.random.rand() + 0.5
omega = np.hstack([x, sig])
log_f_omega = log_f._eval(omega)
print 'Value of log f(x, sigma)'
print '-' * 80

# Evaluate Joint at many points 
k = 0.1 * np.random.rand(10, num_input - 1)
sigma = np.random.rand(10,1) + 0.5
omega2 = np.hstack([k , sigma])
state = log_f.eval_all(omega2)
print 'Value of log f(k, sigma)'
print '-' * 80
print state['log_p']
print 'Gradient'
print '-' * 80
print 'Shape:', len(state['log_p_grad']), 'x', state['log_p_grad'][0].shape
print 'Here is one of them:'
print state['log_p_grad'][0]
print 'Hessian'
print '-' * 80
print 'Shape:', len(state['log_p_grad_2']), 'x', state['log_p_grad_2'][0].shape
print 'Here is one of them:'
print state['log_p_grad_2'][0]
