"""
Solve the inverse problem using the variational methodology.

Author:
    Ilias Bilionis

Date:
    9/10/2014

"""


from collections import Iterable
from common import *
from vuq import *


def parse_expr_callback(option, opt, value, parser):
    """
    This is not very good for security, but it is a fast
    way to pass options.
    """
    if value is not None:
        setattr(parser.values, option.dest, eval(value))


def parse_list_callback(option, opt, value, parser):
    if value is not None:
        value = [float(x) for x in value.split(',')]
        if len(value) == 0:
            value = value[0] 
    setattr(parser.values, option.dest, value)


def convert_to_list(l, n):
    if not isinstance(l, Iterable):
        l = [l] * n
    return l


def initialize_bounds(l, u, n):
    l = convert_to_list(l, n)
    u = convert_to_list(u, n)
    b = tuple((l[i], u[i]) for i in xrange(n))
    return b


def main(options):
    """
    The main function.
    """
    # Load the model
    model = initialize_native_model(options.model)
    # Get the target
    log_p = model['log_p']
    # Get the prior
    log_prior = model['log_prior']
    # Construct the initial approximation
    comp = [MultivariateNormal(log_prior.sample().flatten())
            for i in xrange(options.num_comp)]
    log_q = MixtureOfMultivariateNormals(comp)
    if options.mu_init is not None:
        log_q.mu = options.mu_init
        print str(log_q)
    # The entropy approximation
    entropy = eval(options.entropy_approximation + '()')
    # The expectation functional
    expectation_functional = eval(options.expectation_functional + '(log_p)')
    # The Evidence-Lower-BOund
    elbo = EvidenceLowerBound(entropy, expectation_functional)
    # The optimizer
    optimizer = Optimizer(elbo)
    # The upper and lower bounds for everything
    mu_bounds = initialize_bounds(options.mu_lower_bound, options.mu_upper_bound, log_q.num_dim)
    C_bounds = initialize_bounds(options.C_lower_bound, options.C_upper_bound, log_q.num_dim)
    print 'mu_bounds', mu_bounds
    print 'C_bounds', C_bounds
    # The output file
    output_file = options.output
    if output_file is None:
        output_file = os.path.abspath(os.path.splitext(options.model)[0] + '_'
                                      + 'num_comp=' + str(options.num_comp) + '.pcl')
    # Delete the output file if it exists and you want to force calculations
    if os.path.exists(output_file) and options.force:
        print '-', output_file, 'exists'
        print '- I am removing it'
        os.remove(output_file)
    # If the file exists at this point, then this is not a forced calculation
    # and we should just load it
    if os.path.exists(output_file):
        print '- I am not repeating the calculations'
        with open(output_file, 'rb') as fd:
            results = pickle.load(fd)
            L = results['L']
            log_q = results['log_q']
            nfev = results['nfev']
    else:
        # Do the optimization
        L, nfev = optimizer.optimize(log_q,
                                     tol=options.tol,
                                     max_it=options.max_it,
                                     mu_bounds=mu_bounds,
                                     C_bounds=C_bounds,
                                     full_mu=options.optimize_full_mu)
        print str(elbo)
        # Save the results
        results = {}
        results['L'] = L
        results['log_q'] = log_q
        results['nfev'] = nfev
        with open(output_file, 'wb') as fd:
            pickle.dump(results, fd, protocol=pickle.HIGHEST_PROTOCOL)
    # Print a salary of the statistics
    w = log_q.w
    mu = log_q.mu
    C = log_q.C
    c = np.vstack([np.diag(C[i, :, :]) for i in xrange(log_q.num_comp)])
    x_m = np.mean(w * mu, axis=0)
    # The variance of the mixture can only be approximated via sampling...
    samples = log_q.sample(options.std_samples)
    x_s = np.std(samples, axis=0)
    x_05 = np.percentile(samples, 5, axis=0)
    x_95 = np.percentile(samples, 95, axis=0)
    print '{0:10s} {1:10s} {2:10s}'.format('Parameter', 'Mean', 'Std.')
    print '-' * 32
    for i in xrange(log_q.num_dim):
        print '{0:10s} {1:4.6f} +-{2:2.6f}'.format('x_' + str(i+1), x_m[i], 1.96 * x_s[i])
    for i in xrange(log_q.num_dim):
        print '(%1.3f, %1.3f)' % (x_05[i], x_95[i])
    print 'Number of evaluations:', nfev


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-m', '--model', metavar='FILE', dest='model',
                      help='specify the file containing the native model')
    parser.add_option('--entropy-approximation', type=str, dest='entropy_approximation',
                      default='FirstOrderEntropyApproximation',
                      help='specify the entropy approximation to be used')
    parser.add_option('--expectation-functional', type=str, dest='expectation_functional',
                      default='ThirdOrderExpectationFunctional',
                      help='specify the expectation functional to be used')
    parser.add_option('--num-comp', type=int, dest='num_comp', default=1,
                      help='specify the number of components that you want to use')
    parser.add_option('--mu-lower-bound', type=str, dest='mu_lower_bound', action='callback',
                      callback=parse_expr_callback, default=None,
                      help='specify the lower bound for mu')
    parser.add_option('--mu-upper-bound', type=str, dest='mu_upper_bound', action='callback',
                      callback=parse_expr_callback, default=None,
                      help='specify the upper bound for mu') 
    parser.add_option('--C-lower-bound', type=str, dest='C_lower_bound', action='callback',
                      callback=parse_expr_callback, default=1e-6,
                      help='specify the lower bound for C')
    parser.add_option('--C-upper-bound', type=str, dest='C_upper_bound', action='callback',
                      callback=parse_expr_callback, default=100.,
                      help='specify the upper bound for C')
    parser.add_option('--max-it', type=int, dest='max_it', default=100,
                      help='specify the maximum number of iterations')
    parser.add_option('--tol', type=float, dest='tol', default=1e-2,
                      help='specify the tolerance')
    parser.add_option('--std-samples', type=int, dest='std_samples', default=10000,
                      help='specify the number of samples used in the estimation of the std')
    parser.add_option('--optimize-full-mu', action='store_true', dest='optimize_full_mu',
                      help='optimize all the mus simultaneously')
    parser.add_option('-f', '--force', action='store_true', dest='force',
                      help='force the calculation')
    parser.add_option('-o', '--output', metavar='FILE', dest='output',
                      help='specify the output file')
    parser.add_option('--mu-init', type=str, dest='mu_init', action='callback',
                      callback=parse_expr_callback,
                      help='specify the initial point for mu num_comp x num_dim')
    options, args = parser.parse_args()
    main(options)
