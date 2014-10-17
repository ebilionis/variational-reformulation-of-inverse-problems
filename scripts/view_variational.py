"""
View the details of the variational approximation.

Author:
    Ilias Bilionis

Date:
    9/16/2014

"""


import scipy.stats as stats
import numpy as np

from common import *
from vuq import *


def main(options):
    """
    The main function.
    """
    with open(options.results, 'rb') as fd:
        results = pickle.load(fd)
    for key in results.keys():
        print '=' * 80
        print key, ':'
        print str(results[key])
        print '=' * 80
    
    if options.catalysis is None:
        return

    log_q = results['log_q']
    print 'Median\t95\% Interval'
    for i in xrange(log_q.mu.shape[1]):
        mu = log_q.mu[0, i]
        s = np.sqrt(log_q.C[0, i, i])
        if i == log_q.mu.shape[1] - 1:
            rv = stats.lognorm(s, scale=np.exp(mu))
        else:
            rv = stats.lognorm(s, scale=np.exp(mu)/180.)
        I = rv.interval(0.95)
        print '{0:1.4f} & ({1:1.4f}, {2:1.4f}) \\\\'.format(rv.median(), I[0], I[1])

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-r', '--results', metavar='FILE', dest='results',
                      help='specify the file containing the results of the variational'
                           ' approac')
    parser.add_option('--catalysis', dest='catalysis', action='store_true',
                      help='perform analysis that is related to catalysis')
    options, args = parser.parse_args()
    main(options)
