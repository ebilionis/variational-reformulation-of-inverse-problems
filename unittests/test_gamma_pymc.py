"""
Test if our gamma distribution is exactly the same as pymc's gamma dist.

Author:
    Ilias Bilionis

Date:
    9/9/2014

"""


import numpy as np
import scipy.stats as st
from vuq import GammaPDF
import pymc as pm


a = 4.56
b = 0.5

# Our Gamma
log_p = GammaPDF(a, b, 1)
x = np.random.gamma(a, b, 10)[:, None]

mc_p = pm.Gamma('x', a, 1. / b)

print mc_p.__dict__
mc_log_p = []
for xx in x.flatten():
    mc_p.value = xx
    print '%.4f\t%.4f\t%.4f' % (xx, log_p(np.array([[xx]]))[0], mc_p.logp)
