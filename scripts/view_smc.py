"""
A script that allows you to see the results of an SMC simulation.

Author:
    Ilias Bilionis

Date:
    9/10/2014

"""


from common import *
import pysmc as ps


def main(options):
    """
    The main function.
    """
    db = ps.DataBase.load(options.db_filename)
    if db.gamma < 1.:
        print '*' * 80
        print '- gamma: {0:1.3f}'.format(db.gamma)
        print '- this means that the SMC simulation is not over yet'
        print '- you must resume it'
        print '- I show you the results any way'
        print '*' * 80
    pa = db.particle_approximation
    w = pa.weights
    print 'Summary of SMC Statistics'.center(32)
    print '=' * 32
    print '{0:10s} {1:10s} {2:10s}'.format('Parameter'.center(10),
                                           'Mean'.center(10),
                                           'Std.'.center(10))
    print '-' * 32
    for name in pa.stochastics.keys():
        x = pa.stochastics[name]
        x_m = np.sum((w * x.T).T, axis=0)
        x2_m = np.sum((w * (x ** 2.).T).T, axis=0)
        x_s = np.sqrt(x2_m - x_m ** 2.)
        if isinstance(x_m, np.ndarray):
            for i in xrange(x_m.shape[0]):
                print '{0:10s} {1:4.6f} +-{2:2.6f}'.format(name + '_' + str(i+1),
                                                           x_m[i], x_s[i]).center(32)
        else:
            print '{0:10s} {1:4.6f} +-{2:2.6f}'.format(name, x_m, x_s).center(32)
    print '=' * 32

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--db-filename', metavar='FILE', dest='db_filename',
                      help='specify the SMC database you want to look at')
    options, args = parser.parse_args()
    main(options)
