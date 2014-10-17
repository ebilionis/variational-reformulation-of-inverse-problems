"""
A script that allows you to see the results of an MCMC simulation.

Author:
    Ilias Bilionis

Date:
    9/11/2014

"""


from common import *
import tables as tb
import matplotlib.pyplot as plt


def main(options):
    """
    The main function.
    """
    f = tb.open_file(options.db_filename, mode='r')
    chain = options.chain
    if chain is None:
        chain_counter = f.get_node('/mcmc/chain_counter')
        chain = chain_counter[-1][-1]
    chain_data = getattr(f.root.mcmc.data, chain)
    count = chain_data.cols.step[options.skip::options.thin]
    accepted = chain_data.cols.accepted[options.skip::options.thin]
    acceptance_rate = np.array(accepted, dtype='float32') / count
    data = chain_data.cols.params[options.skip::options.thin]
    plt.hist(np.exp(data[:, 0]))
    plt.savefig('test_hist.png')
    x_m = np.mean(data, axis=0)[None, :]
    x_s = np.std(data, axis=0)[None, :]
    x_05 = np.percentile(data, 5, axis=0)[None, :]
    x_95 = np.percentile(data, 95, axis=0)[None, :]
    f.close()
    if x_m.shape[1] < 8:
        width = x_m.shape[1] * 10 + 20
        print 'Running Mean'.center(width)
        print '=' * width
        names = ['{0:10s}'.format('x_' + str(i + 1)) for i in xrange(x_m.shape[1])]
        print '{0:10s} {1:10s}'.format('Step', 'Acc. Rate') + ' '.join(names)
        for i in xrange(x_m.shape[0]):
            vals = ['{0:1.6f}'.format(x_m[i, j]) + ' ' for j in xrange(x_m.shape[1])]
            print '{0:10s} {1:1.3f}'.format(str(count[i]), acceptance_rate[i]) + ' ' * 6 + ' '.join(vals)
        print '=' * width
        print 'Running Std'.center(width)
        print '=' * width
        names = ['{0:10s}'.format('x_' + str(i + 1)) for i in xrange(x_m.shape[1])]
        print '{0:10s} {1:10s}'.format('Step', 'Acc. Rate') + ' '.join(names)
        for i in xrange(x_m.shape[0]):
            vals = ['{0:1.6f}'.format(2. * x_s[i, j]) + ' ' for j in xrange(x_m.shape[1])]
            print '{0:10s} {1:1.3f}'.format(str(count[i]), acceptance_rate[i]) + ' ' * 6 + ' '.join(vals)
        print '=' * width
        for i in xrange(x_m.shape[0]):
            for j in xrange(x_m.shape[1]):
                print '(%1.3f, %1.3f)' % (x_05[i, j], x_95[i, j])


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--db-filename', metavar='FILE', dest='db_filename',
                      help='specify the MCMC database you want to look at')
    parser.add_option('-c', '--chain', type=str, dest='chain',
                      help='specify the MCMC chain you want to look at (default looks at last)')
    parser.add_option('-s', '--skip', type=int, dest='skip', default=0,
                      help='the number of MCMC chain records you want to skip')
    parser.add_option('--thin', type=int, dest='thin', default=1,
                      help='specify how much you want to thin the MCMC chain.')
    options, args = parser.parse_args()
    main(options)
