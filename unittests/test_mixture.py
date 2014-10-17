"""
Test the mixturePDF class.

Author:
    Panagiotis Tsilifis

Date:
    6/5/2014

"""

import numpy as np
import matplotlib.pyplot as plt
from vuq import MultivariateT
from vuq import MixturePDF


# Set the dimension 
num_dim = 2

# Degrees of freedom 
df = 4

mu1 = np.random.standard_t(2, num_dim)
mu2 = np.random.standard_t(4, num_dim)

comp = [MultivariateT(mu1, df), MultivariateT(mu2, df)]

log_q = MixturePDF(comp)

print str(log_q) + '\n'

h = 5
x1 = np.linspace((mu1[0] + mu2[0]) / 2 - h, (mu1[0] + mu2[0]) / 2 + h, 60)
x2 = np.linspace((mu1[1] + mu2[1]) / 2 - h, (mu1[1] + mu2[1]) / 2 + h, 60)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros(X1.shape)
for i in xrange(Z.shape[0]):
    for j in xrange(Z.shape[1]):
        x0 = np.array([X1[i,j], X2[i,j]])
        Z[i,j] = log_q._eval(x0)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.contourf(X1, X2, Z, levels=np.linspace(-8., -1., 50))
ax.set_xlabel('$x1$', fontsize=16)
ax.set_ylabel('$x2$', fontsize=16)
cbar = fig.colorbar(cax)
plt.show()

#Evaluate the gradient at x
x = np.random.standard_t(3, [10,2])
grad_log_q = log_q.grad(x)
hess_log_q = log_q.hessian(x)
print 'Evaluating gradient at x'# + str(x) 
print '-' * 80
print 'Shape:', len(grad_log_q), 'x', grad_log_q[0].shape
print 'Evaluating hessian at x'# + str(x)
print '-' * 80
print 'Shape:', len(hess_log_q), 'x', hess_log_q[0].shape
