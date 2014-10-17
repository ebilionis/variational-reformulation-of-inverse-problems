"""
Test how the likelihood functions work.

Author:
    Panagiotis Tsilifis

Date:
    5/26/2014

"""

import numpy as np
from vuq import IndepNoiseGaussianLikelihood
from vuq import IsotropicGaussianLikelihood

import sys
sys.path.insert(0,'demos/')

from catalysis import CatalysisModelDMNLESS
from diffusion import ContaminantTransportModel

# Define the data first
data = np.loadtxt('data.txt').reshape((7, 6))
y = data[:, 1:]
y = y.reshape((1,y.shape[0] * y.shape[1]))
# Define the model
num_dim = 5
catal_model = CatalysisModelDMNLESS()

# Define the isotropic Likelihood
IsotropicL = IsotropicGaussianLikelihood(y[0,:] / 500, catal_model)

print str(IsotropicL) + '\n'

kappa = np.random.rand(5)
values = catal_model._eval(kappa)
fx = values['f']

# Test the likelihood results
Iso_state = IsotropicL._noise_eval(fx,np.array([2.]))
print 'Likelihood value'
print '-' * 80
print str(Iso_state['L']) + '\n'
print 'dL/dfx'
print '-' * 80
print 'Shape : ' + str(Iso_state['L_grad_f'].shape) + '\n'
print str(Iso_state['L_grad_f']) + '\n'
print 'dL/dtheta'
print '-' * 80
print 'Shape : ' + str(Iso_state['L_grad_theta'].shape) + '\n'
print str(Iso_state['L_grad_theta']) + '\n'
print 'd2L/dfx2'
print '-' * 80
print 'Shape : ' + str(Iso_state['L_grad_2_f'].shape) + '\n'
print str(Iso_state['L_grad_2_f']) + '\n'
print 'd2L/dtheta2'
print '-' * 80
print 'Shape : ' + str(Iso_state['L_grad_2_theta'].shape) + '\n'
print str(Iso_state['L_grad_2_theta']) + '\n'
print 'd2L/dtheta_df'
print '-' * 80
print 'Shape : ' + str(Iso_state['L_grad_2_theta_f'].shape) + '\n'
print str(Iso_state['L_grad_2_theta_f']) + '\n'

# Define the Independent noises Likelihood
IndepNoiseL = IndepNoiseGaussianLikelihood(y[0,:], catal_model)

print str(IndepNoiseL) + '\n'

# Test the likelihood results
Indep_state = IndepNoiseL._noise_eval(fx,2.*np.ones(fx.shape[0]))
print 'Likelihood value'
print '-' * 80
print str(Indep_state['L']) + '\n'
print 'dL/dfx'
print '-' * 80
print 'Shape : ' + str(Indep_state['L_grad_f'].shape) + '\n'
print str(Indep_state['L_grad_f']) + '\n'
print 'dL/dtheta'
print '-' * 80
print 'Shape : ' + str(Indep_state['L_grad_theta'].shape) + '\n'
print str(Indep_state['L_grad_theta']) + '\n'
print str(sum(Indep_state['L_grad_theta'])) + '\n' # For equal parameters this must equal the isotropic dLdtheta
print 'd2L/dfx2'
print '-' * 80
print 'Shape : ' + str(Indep_state['L_grad_2_f'].shape) + '\n'
print str(Indep_state['L_grad_2_f']) + '\n'
print 'd2L/dtheta2'
print '-' * 80
print 'Shape : ' + str(Indep_state['L_grad_2_theta'].shape) + '\n'
print str(Indep_state['L_grad_2_theta']) + '\n'
print str(sum(np.diag(Indep_state['L_grad_2_theta'])))  + '\n'# For equal parameters this must equal the isotropic d2Ldtheta2
print 'd2L/dtheta_dfx'
print '-' * 80
print 'Shape : ' + str(Indep_state['L_grad_2_theta_f'].shape) + '\n'
print str(Indep_state['L_grad_2_theta_f'])

# Test the likelihood function on the diffusion model 
data_diff = np.load('data_concentrations.npy')
# The model
diff_model = ContaminantTransportModel()
# Define the isotropic Likelihood
IsotropicL_diff = IsotropicGaussianLikelihood(data_diff[:], diff_model)

print str(IsotropicL_diff) + '\n'

xs = np.random.rand(2)
values_x = diff_model._eval(xs)
fxs = values_x['f']
# Test the likelihood results
Iso_state_diff = IsotropicL_diff._noise_eval(fxs,np.array([2.]))
print 'Likelihood value'
print '-' * 80
print str(Iso_state_diff['L']) + '\n'
print 'dL/dfx'
print '-' * 80
print 'Shape : ' + str(Iso_state_diff['L_grad_f'].shape) + '\n'
print str(Iso_state_diff['L_grad_f']) + '\n'