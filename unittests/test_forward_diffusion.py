"""
Test the diffusion forward model.

Author:
    Panagiotis Tsilifis

Date:
    6/12/2014

"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import fipy as fp
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'./demos/')
from diffusion import ContaminantTransportModel


data = np.load('data_concentrations.npy')
xs = np.array([0.09, 0.23])
model = ContaminantTransportModel()
state = model._eval(xs)
print 'Data\t\tForward model\t\tAbs. Difference'
print '-' * 80
#print state['f']
for i in range(16):
    print '{0:7f}\t\t{1:7f}\t\t{2:7f}'.format(data[i], state['f'][i], np.abs(data[i] - state['f'][i]))
    
dU = state['f_grad']
print 'Derivatives data wrt x_s1 and xs_2:'
print '-' * 80
print 'Shape : '
print str(dU.shape) + '\n'
print str(dU) + '\n'

d2U = state['f_grad_2']
print '2nd derivatives data wrt x_s1 :'
print '-' * 80
print 'Shape : ' 
print str(d2U.shape) + '\n'
print str(d2U) + '\n'
