"""
Show some results for SMC.

Author:
    Ilias Bilionis

Date:
    6/26/2014

"""


import matplotlib
matplotlib.use('GTK')
import numpy as np
import pysmc as ps
import matplotlib.pyplot as plt


db = ps.DataBase.load('demos/smc_catalysis.pcl')
p = db.particle_approximation
w = p.weights
sig = p.sigma2
plt.hist(sig, weights=w)
plt.show()
kappa = p.kappa
for i in xrange(kappa.shape[1]):
    plt.hist(kappa[:, i], weights=w)
    plt.show()
