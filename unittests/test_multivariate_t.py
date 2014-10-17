"""
Test the multivariate student-t class.

Author:
    Panagiotis Tsilifis

Date:
    6/5/2014

"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from vuq import MultivariateT


# The number of dimensions
num_dim = 1

# Degrees of freedom
df = 10

# Location parameter
mu = np.array([0.])

# The input pdf
log_p = MultivariateT(mu,df)

# Print the object
print str(log_p) + '\n'

# Evaluate the distribution at some random points
# and compare to what we get from scipy
print 'Our Student-t\t\tScipy Student-t\t\tAbs. Difference'
print '-' * 80
for i in xrange(10):
    x = np.random.standard_t(df,1)
    sval = st.t.logpdf(x, df) 
    val = log_p._eval(x) 
    print '{0:7f}\t\t{1:7f}\t\t{2:7f}'.format(val, sval[0], np.abs(val - sval[0]))
print '-' * 80 + '\n'

df2 = 1e+4
N = 10

# Test the evaluation at many inputs simultaneously and compare with standard normal for high df 
print 'Evaluating at many points'
print '-' * 80
x = np.random.standard_t(df, [N,1])
y = log_p(x)
print 'Shape:', y.shape
print 'log pdf:'
print str(y) + '\n'
print 'Our Student-t (df = 1e+4)\t\tScipy Student\t\tScipy Normal\t\tAbs. Difference'
for i in range(N):
    sval = st.t.logpdf(x[i,0], df2)
    sval2 = st.norm.logpdf(x[i,0])
    val = log_p._eval(x[i,:])
    print '{0:7f}\t\t{1:7f}\t\t{2:7f}\t\t{3:3f}'.format(val, sval, sval2, np.abs(val - sval2))
    
# Test accuracy wrt df 
#df3 = np.arange(1, 1e+5, 100)
#x = np.random.standard_t(df, 1)
#sval = np.array([st.t.logpdf(x, df3[i]) for i in range(df3.shape[0])])
#val = np.zeros(df3.shape[0])
#for i in range(df3.shape[0]):
#    log_p = MultivariateT(mu, df3[i])
#    val[i] = log_p._eval(x)
#plt.plot(df3, sval - val)
#plt.show()

mu0 = np.random.standard_t(2, 5)
log_p0 = MultivariateT(mu0, df)
print str(log_p0)
x0 = np.random.standard_t(df, [N,5])
# Test the evaluation of the gradient at many inputs simultaneously
print 'Evaluating gradient at many points'
print '-' * 80
dy = log_p0.grad(x0)
print 'Shape:', dy.shape
print 'grad log pdf:'
print str(dy) + '\n'

# Test the evaluation of the Hessian at many inputs simultaneously
print 'Evaluating the Hessian at many points'
print '-' * 80
hy = log_p0.hessian(x0)
print 'Shape:', hy.shape
print 'hessian log pdf:'
print str(hy) + '\n'