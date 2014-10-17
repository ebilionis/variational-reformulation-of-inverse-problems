"""
Test how the collection works.

Author:
    Ilias Bilionis

Date:
    5/22/2014

"""


import numpy as np
import scipy.stats as st
from vuq import PDFCollection
from vuq import GammaPDF
from vuq import UniformND


# The number of dimensions
num_input = 10

# The parameters
# Alpha is chosen uniformly from [0,10]
alpha = 10*np.random.rand(num_input)
# Beta is chosen uniformly from [0,1]
beta = np.random.rand(num_input)

# Make a collection of Gammas:
#log_p = PDFCollection([ GammaPDF(alpha[i], beta[i], 1) for i in xrange(num_input) ])
log_p = PDFCollection([ UniformND(1) for i in xrange(num_input) ])

print str(log_p)

print 'Evaluating at many points'
print '-' * 80
#x = np.random.gamma(10, 1, [20, num_input])
x = np.random.rand(20, num_input)
y = log_p(x)
print 'Shape:', y.shape
print 'log pdf:'
print str(y) + '\n'


# Test the evaluation of the gradient at many inputs simultaneously
print 'Evaluating gradient at many points'
print '-' * 80
dy = log_p.grad(x)
print 'Shape:', len(dy), 'x', dy[0].shape
#print 'grad log pdf:'
#print str(dy) + '\n'

# Test the evaluation of the Hessian at many inputs simultaneously
print 'Evaluating the Hessian at many points'
print '-' * 80
d2y = log_p.hessian(x)
print 'Shape:', len(d2y)
print d2y[0]

# Test the evaluation of everything simultaneously
print 'Evaluating everything (log_p, grad and Hessian) at all points'
state = log_p.eval_all(x)
print 'I god back a', type(state), 'with keys:', state.keys()
print 'I am looping over the keys:'
for key in state.keys():
    print key, 'is a', type(state[key])

