"""
Test the catalysis forward model.

Author:
    Panagiotis Tsilifis

Date:
    5/22/2014

"""

import numpy as np
from matplotlib.pyplot import *
import sys
sys.path.insert(0,'./demos/')
from catalysis import CatalysisModel
from catalysis import CatalysisModelDMNLESS
from catalysis import CatalysisFullModelDMNLESS


# The data 
data = np.loadtxt('data.txt').reshape((7, 6))
t = data[:, 0]
y_obs = data[:, 1:]
idx = [0, 1, 5, 2, 3, 4]
y_obs = np.hstack([y_obs, np.zeros((y_obs.shape[0], 1))])
y_obs = y_obs[:, idx]

x = np.array([0.0216, 0.0292, 0.0219, 0.0021, 0.0048])

# Test first the original model 
solution = CatalysisModel()

print str(solution) + '\n'

state = solution._eval(x)
y = state['f']
J = state['f_grad']
H = state['f_grad_2']
print 'Solution'
print '-' * 80
print 'Shape : ' + str(y.shape)
print y.reshape((7, y.shape[0]/7))
print '\n'
print 'Jacobian'
print '-' * 80
print J
print '\n'
print 'Second derivatives'
print '-' * 80
print H
#t = np.array([0.0, 30., 60., 90., 120., 150., 180.])
plot(t, y_obs, '*')
plot(t, y.reshape((t.shape[0], y.shape[0]/t.shape[0])))
show()



print 'Test the evaluation at many inputs simultaneously'
kappa = 0.1*np.random.rand(x.shape[0],5)
state2 = solution(kappa)
print 'Solution'
print '-' * 80
print str(state['f']) + '\n'
print 'First derivates'
print '-' * 80
print str(state['f_grad']) + '\n'
print 'Second derivatives'
print '-' * 80
print str(state['f_grad_2']) + '\n'

# Now test the dimensionless version of the model
solution_dmnl = CatalysisModelDMNLESS()

print str(solution_dmnl) + '\n'

state2 = solution_dmnl._eval(x * 180)
y2 = state2['f']
J2 = state2['f_grad']
H2 = state2['f_grad_2']
print 'Solution'
print '-' * 80
print y2.reshape((7, y2.shape[0]/7))
print '\n'
print 'Jacobian'
print '-' * 80
print J2
print '\n'
print 'Second derivatives'
print '-' * 80
print H2
#t = np.array([0.0, 1./6, 1./3, 1./2, 2./3, 5./6, 1.])
plot(t/180., y_obs/500., '*')
plot(t/180., y2.reshape((t.shape[0], y2.shape[0]/t.shape[0])))
show()

print H.shape
#plot(J.flatten() / 500 / 180, J2.flatten(), '.')
plot(H.flatten() / 500 / 180 ** 2, H2.flatten(), '.')
show()
quit()

x_full = np.array([[-3.888,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
       [ 3.888, -6.498,  0.   ,  0.   ,  0.   ,  0.   ],
       [ 0.   ,  5.256, -3.942,  0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.   ,  3.942,  0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.864,  0.   ,  0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.378,  0.   ,  0.   ,  0.   ,  0.   ]])
x_full = x_full.reshape(36)

# Now test the dimensionless version of the full model
solution_full = CatalysisFullModelDMNLESS()

print str(solution_full) + '\n'

state3 = solution_full._eval(x_full)
y3 = state3['f']
J3 = state3['f_grad']
H3 = state3['f_grad_2']
print 'Solution'
print '-' * 80
print y3.reshape((7 ,y3.shape[0] / 7))
print '\n'
print 'Jacobian'
print '-' * 80
print J3
print '\n'
print 'Second derivative'
print '-' * 80
print H3
plot(t/180, y_obs/500., '*')
plot(t/180., y3.reshape((t.shape[0], y3.shape[0]/t.shape[0])))
show()

x_full_numer = np.array([[-4.33196874, -5.        ,  6.67939317, -1.53947333,  3.98937717,
         6.15844821],
       [ 2.78059767, -5.        ,  1.74400916, -0.21704501, -3.52037185,
        -0.53811489],
       [ 1.60431762,  2.6435434 , -4.99999968, -0.67762272, -2.85725542,
         0.75467099],
       [-0.51638391,  1.45563688,  3.67548823,  1.56088914, -0.25853257,
        -5.        ],
       [ 0.37593609, -3.80254408,  4.2799548 ,  1.38250366, -2.8156011 ,
        -5.        ],
       [ 0.24968126, -4.61529558,  5.73391027, -0.50962955,  1.67635654,
         2.73356322]])
x_full_numer = x_full_numer.reshape(36)
# Now test the dimensionless version of the full model

state4 = solution_full._eval(x_full_numer)
y4 = state4['f']
J4 = state4['f_grad']
H4 = state4['f_grad_2']
print 'Solution'
print '-' * 80
print y4.reshape((7 ,y4.shape[0] / 7))
print '\n'
print 'Jacobian'
print '-' * 80
print J4
print '\n'
print 'Second derivative'
print '-' * 80
print H4
plot(t/180, y_obs/500., '*')
plot(t/180., y4.reshape((t.shape[0], y4.shape[0]/t.shape[0])))
show()

"""
"""
A = np.array([[-3.76978464, -0.50424625, -0.21679456, 0.29876246, -0.03181339, 0.03879548],
[ 2.41461854, -1.22813897, -2.20618065, 0.00794199, 0.36123171, -0.00530061],
[ 1.58491466, -0.0048723, -0.36971964, -0.62944618, 0.0174473, -0.12462298],
[-0.39650543, 1.6740821, 2.42152955, 0.08367671, -0.28383747, 0.04730487],
[ 0.14612687, -0.217782, -0.02277859, 0.19942222, -0.02387345, 0.02980567],
[ 0.02063, 0.28095741, 0.39394389, 0.0396428, -0.03915471, 0.01401757]])

A = A.reshape(36)

state5 = solution_full._eval(A)
y5 = state5['f']
J5 = state5['f_grad']
H5 = state5['f_grad_2']
plot(t/180, y_obs/500., '*')
plot(t/180., y5.reshape((t.shape[0], y5.shape[0]/t.shape[0])))
show()

B = np.zeros(36)
for i in xrange(36):
    if A[i] > 0.1:
        B[i] = A[i]

print 'Matrix' + '\n'
print str(B)
state6 = solution_full._eval(A)
y6 = state6['f']
J6 = state6['f_grad']
H6 = state6['f_grad_2']
plot(t/180, y_obs/500., '*')
plot(t/180., y6.reshape((t.shape[0], y6.shape[0]/t.shape[0])))
show()
