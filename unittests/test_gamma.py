"""
Test the multivariate Gamma class.

Author:
    Panagiotis Tsilifis

Date:
    5/21/2014

"""


import numpy as np
import scipy.stats as st
from vuq import GammaPDF


# The number of dimensions
num_input = 10

# The parameters
# Alpha is chosen uniformly from [0,10]
alpha = 10*np.random.rand(1)
# Beta is chosen uniformly from [0,1]
beta = np.random.rand(1)

# The input pdf
log_p = GammaPDF(alpha,beta,num_input)

# Print the object
print str(log_p) + '\n'

# Evaluate the distribution at some random points
# and compare to what we get from scipy
print 'Our Gamma\t\tScipy Gamma\t\tAbs. Difference'
print '-' * 80
for i in range(10):
    x = np.random.gamma(10, 1, num_input)
    sval = np.sum([st.gamma.logpdf(x[i], alpha, scale = beta) for i in xrange(num_input)])
    val = log_p._eval(x)
    print '{0:7f}\t\t{1:7f}\t\t{2:7f}'.format(val[0], sval, np.abs(val[0] - sval))
print '-' * 80 + '\n'

# Test the evaluation at many inputs simultaneously
print 'Evaluating at many points'
print '-' * 80
x = np.random.gamma(10, 1,[3, num_input])
y = log_p(x)
print 'Shape:', y.shape
print 'log pdf:'
print str(y) + '\n'

# Test the evaluation of the gradient at many inputs simultaneously
print 'Evaluating gradient at many points'
print '-' * 80
dy = log_p.grad(x)
print 'Shape:', dy.shape
print 'grad log pdf:'
print str(dy) + '\n'

# Test the evaluation of the Hessian at many inputs simultaneously
print 'Evaluating the Hessian at many points'
print '-' * 80
hy = log_p.hessian(x)
print 'Shape:', hy.shape
print 'hessian log pdf:'
print str(hy) + '\n'