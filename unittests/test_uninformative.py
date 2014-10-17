"""
Test the multivariate uninformative class.

Author:
    Panagiotis Tsilifis

Date:
    5/21/2014

"""


import numpy as np
import scipy.stats as st
from vuq import UninformativePDF


# The number of dimensions
num_input = 10

# The input pdf
log_p = UninformativePDF(num_input)

# Print the object
print str(log_p) + '\n'

# Re-compute the distribution at some random points
# and compare to what we get from our UninformativePDF
print 'x\t\tlogp(x)\t\tAbs. Difference'
print '-' * 80
for i in range(10):
    x = np.random.exponential(1, num_input)
    sval = np.sum([-np.log(x[i]) for i in xrange(num_input)])
    val = log_p._eval(x)
    print '{0:1f}\t\t{1:1f}\t\t{2:1f}'.format(val, sval, np.abs(val - sval))
print '-' * 80 + '\n'

# Test the evaluation at many inputs simultaneously
print 'Evaluating at many points'
print '-' * 80
x = np.random.exponential(1,[3,num_input])
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