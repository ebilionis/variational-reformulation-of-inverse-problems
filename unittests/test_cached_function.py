"""
Tests the vuq.CachedFunction decorator.

Author:
    Ilias Bilionis

Date:
    6/6/2014

"""


import numpy as np
import time
from vuq import *


# Test the evaluation of a cached and of an un-cached function

# An expensive function
def f(x):
    for i in xrange(1000):
        x += i
    x /= 1000000
    return 2. * x


#@CachedFunction()
def f_c(x):
    return f(x)

class Foo(object):

    def __init__(self):
        self._f = f

    def __call__(self, x):
        return self._f(x)

class FooCached(object):

    def __init__(self):
        self._f = f

    def __call__(self, x):
        return self._f(x)


# A set of points to evaluate the function repeatedly
X = np.random.rand(10, 10)
# How many repetitions to do
num_repeat = 1000

F = Foo()
Fc = CachedFunction(F.__call__)

t0 = time.time()
for i in xrange(num_repeat):
    y = F(X[i % X.shape[0], :])
t1 = time.time()
print 'Elapsed time without cache (seconds):', t1 - t0

t0 = time.time()
for i in xrange(num_repeat):
    y = Fc(X[i % X.shape[0], :])
t1 = time.time()
print 'Elapsed time with cache (seconds):', t1 - t0
