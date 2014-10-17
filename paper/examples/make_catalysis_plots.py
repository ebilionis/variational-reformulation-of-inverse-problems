"""
A script that processes the catalysis results and plots a few things.

Author:
    Ilias Bilionis

Date:
    9/9/2014

"""


from optparse import OptionParser
from catalysis_common import *
import catalysis_dimensionless_pymc_model
import catalysis_dimensionless_model
import matplotlib.pyplot as plt
import cPickle as pickle
import tables as tb
import pandas as pd
import pysmc as ps
from scipy import stats
from vuq import MultivariateNormal
from vuq import MixtureOfMultivariateNormals


def main(options):
    """
    The main function.
    """
    # DATA AND MODEL
    data = load_dimensionless_catalysis_data()
    model = catalysis_dimensionless_model.make_model()
    catal_model = model['catal_model']
    # VARIATIONAL PLOTS
    with open(options.var_file, 'rb') as fd:
        var_results = pickle.load(fd)
    log_q = var_results['log_q']
    if var_results.has_key('output_samples') and not options.force:
        Y = var_results['output_samples']
    else:
        Y = []
        for i in xrange(options.num_samples):
            print 'taking sample', i + 1
            omega = log_q.sample().flatten()
            x = omega[:5]
            sigma = np.exp(omega[5])
            y = catal_model(x)['f']
            Y.append(y + sigma * np.random.randn(*y.shape))
        Y = np.vstack(Y)
        var_results['output_samples'] = Y
        with open(options.var_file, 'wb') as fd:
            pickle.dump(var_results, fd, protocol=pickle.HIGHEST_PROTOCOL)
    var_out_fig = os.path.abspath(os.path.splitext(options.var_file)[0] + '_output.png')
    plot_catalysis_output(var_out_fig, Y, legend=True, title='VAR. (L=%d)' % log_q.num_comp)
    # MCMC PLOTS
    f = tb.open_file(options.mcmc_file, mode='r')
    chain = options.chain
    if chain is None:
        chain_counter = f.get_node('/mcmc/chain_counter')
        chain = chain_counter[-1][-1]
    chain_data = getattr(f.root.mcmc.data, chain)
    mcmc_step = chain_data.cols.step[:]
    omega = chain_data.cols.params[:]
    x = omega[:, :5]
    sigma = np.exp(omega[:, 5])
    # Do not do the following step unless this is a forced calculation
    mcmc_file_prefix = os.path.abspath(os.path.splitext(options.mcmc_file)[0]) +\
                                       '_ch=' + str(chain)
    mcmc_output_file = mcmc_file_prefix + '_output.pcl'
    if os.path.exists(mcmc_output_file) and not options.force:
        print '- I found', mcmc_output_file
        with open(mcmc_output_file, 'rb') as fd:
            Y_mcmc = pickle.load(fd)
    elif not options.skip_mcmc_output:
        print '- computing the output of the model for each one of the MCMC samples'
        print '- be patient'
        Y_mcmc = catal_model(x)['f']
        with open(mcmc_output_file, 'wb') as fd:
            pickle.dump(Y_mcmc, fd, protocol=pickle.HIGHEST_PROTOCOL)
    if not options.skip_mcmc_output:
        Y_mcmc = Y_mcmc[options.skip::options.thin, :]
        sigma = sigma[options.skip::options.thin]
        Y_mcmc += sigma[:, None] * np.random.randn(*Y_mcmc.shape)
        mcmc_out_fig = mcmc_file_prefix + '_output.png'
        plot_catalysis_output(mcmc_out_fig, Y_mcmc, title='MCMC (MALA)')
    # Plot the histograms
    or_omega = omega
    omega = np.exp(omega[options.skip::options.thin, :])
    w = log_q.w
    mu =log_q.mu
    C = log_q.C
    c = np.vstack([np.diag(C[i, :, :]) for i in xrange(log_q.num_comp)])
    for i in xrange(omega.shape[1]):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(omega[:, i], alpha=0.5, normed=True)
        #ax.hist(smc_samples[:, i], weights=smc_weights, alpha=0.5, normed=True)
        x_min = omega[:, i].min()
        x_max = omega[:, i].max()
        x_i = np.linspace(x_min, x_max, 100)
        # Create a 1D mixture
        comp = [MultivariateNormal([[mu[j, i]]], C=[[c[j, i]]])
                for j in xrange(log_q.num_comp)]
        log_q_i = MixtureOfMultivariateNormals(comp)
        y_i = np.exp(log_q_i(np.log(x_i[:, None]))) / x_i
        mu_i = 0. if i < 5 else -1.
        x_p_max = min(np.exp(mu_i + 2.), x_i[-1])
        x_p_i = np.linspace(np.exp(mu_i - 2.), x_p_max, 100)
        ax.plot(x_p_i, stats.norm.pdf(np.log(x_p_i), mu_i, scale=1.) / x_p_i,
                'g--', linewidth=2)
        ax.plot(x_i, y_i, 'r-', linewidth=2)
        #ax.hist(samples_var[:, i], alpha=0.5, normed=True)
        name  = '\kappa_%d' % (i + 1) if i < 5 else '\sigma'
        xlabel = '$' + name + '$'
        ylabel = '$p(' + name + '|y)$'
        ax.set_xlabel(xlabel, fontsize=26)
        ax.set_ylabel(ylabel, fontsize=26)
        plt.setp(ax.get_xticklabels(), fontsize=26)
        plt.setp(ax.get_yticklabels(), fontsize=26)
        if i == 0:
            leg = plt.legend(['Prior', 'VAR ($L=%d$)' % log_q.num_comp, 'MCMC (MALA)'], loc='best')
            plt.setp(leg.get_texts(), fontsize=26)
        if i == 5:
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.tight_layout()
        png_file = os.path.abspath(os.path.splitext(options.var_file)[0] +
                                   '_input_' + str(i) + '.png')
        print '- writing', png_file
        plt.savefig(png_file)
        del fig
    # Draw the error of MCMC as a function of the number of evaluations
    # Compute the MCMC mean
    #x_m_mcmc = pd.rolling_mean(omega, options.rolling)
    #x_s_mcmc = pd.rolling_std(omega, options.rolling)
    omega = or_omega[50:, :]
    mcmc_step = mcmc_step[50:]
    x_m_mcmc = np.cumsum(omega, axis=0) / np.arange(1, omega.shape[0] + 1)[:, None]
    x_m_mcmc_2 = np.cumsum(omega ** 2., axis=0) / np.arange(1, omega.shape[0] + 1)[:, None]
    x_s_mcmc = np.sqrt(x_m_mcmc_2 - x_m_mcmc ** 2.) 
    x_m_var = np.mean(w * mu, axis=0)
    samples_var = log_q.sample(options.var_samples)
    x_s_var = np.std(samples_var, axis=0)
    # We will compute the mean and the std of MCMC as accurately as possible
    t_m = np.mean(or_omega[700:, :], axis=0)
    t_s = np.std(or_omega[700:, :], axis=0)
    #t_m = x_m_mcmc[-1, :]
    #t_s = x_s_mcmc[-1, :]
    e_m = x_m_mcmc - t_m
    m_rms = np.sqrt(np.mean(e_m ** 2., axis=1)) / np.linalg.norm(t_m)
    e_s = x_s_mcmc - t_s
    s_rms = np.sqrt(np.mean(e_s ** 2., axis=1)) / np.linalg.norm(t_s)
    # Variational error
    v_m_rms = np.linalg.norm(mu - t_m) / np.linalg.norm(t_m)
    v_s_rms = np.linalg.norm(np.sqrt(c) - t_s) / np.linalg.norm(t_s)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(mcmc_step, m_rms, '-b', linewidth=2)
    ax.plot(mcmc_step, np.ones(mcmc_step.shape[0]) * v_m_rms, '--b', linewidth=2)
    ax.plot(mcmc_step, s_rms, '.-g', linewidth=2)
    ax.plot(mcmc_step, np.ones(mcmc_step.shape[0]) * v_s_rms, ':g', linewidth=2)
    ax.set_xlim([0, 70000])
    ax.set_yscale('log')
    png_file = 'rms.png'
    plt.savefig(png_file)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--var-file', dest='var_file', metavar='FILE',
                      help='set the variational results file')
    parser.add_option('--mcmc-file', dest='mcmc_file', metavar='FILE',
                      help='set the input file (result of an MCMC simulation)')
    parser.add_option('--num-samples', dest='num_samples', type=int, default=10000,
                      help='the number of samples used to compute the error bars')
    parser.add_option('-f', '--force', dest='force', action='store_true',
                      help='force the calculations')
    parser.add_option('--skip-mcmc-output', dest='skip_mcmc_output', action='store_true',
                      help='skip the expensive plots that involves computation of MCMC output')
    parser.add_option('-c', '--chain', type=str, dest='chain',
                      help='specify the MCMC chain you want to look at (default looks at last)')
    parser.add_option('-s', '--skip', type=int, dest='skip', default=100,
                      help='the number of MCMC chain records you want to skip')
    parser.add_option('--thin', type=int, dest='thin', default=1,
                      help='specify how much you want to thin the MCMC chain.')
    parser.add_option('--max', type=int, dest='max', default=-1,
                      help='specify the maximum mcmc step you want to use')
    parser.add_option('--var-samples', type=int, dest='var_samples', default=10000,
                      help='the number of samples used to compute the standard deviation of'
                           ' log_q')
    parser.add_option('--rolling', type=int, dest='rolling', default=100,
                      help='specify the number of samples used to compute the rolling mean')
    options, args = parser.parse_args()
    main(options)
