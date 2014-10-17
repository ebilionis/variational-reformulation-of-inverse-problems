"""
Does MCMC with a PyMC model, but uses MALA.

MALA = Metropolis-Adjusted-Langevin

Author:
    Ilias Bilionis

Date:
    9/9/2014

"""


from common import *
import pymcmc as pm


class NativeModel(pm.Model):

    """
    A PyMCMC wrapper for the native model.
    """

    def __init__(self, model, name='Native model wrapper'):
        """
        Initialize the object.
        """
        self.model = model
        super(NativeModel, self).__init__(name=name)
        self._state = {}
        self._state['params'] = model['log_prior'].sample().flatten()
        self._eval_state()


    def _eval_state(self):
        """
        Evaluates the state of the model in order to avoid redundant calculations.
        """
        x = self._state['params']
        l = self.model['log_like'](x)
        self._state['log_likelihood'] = l['L'][0]
        self._state['grad_log_likelihood'] = l['L_grad'][0].flatten()
        self._state['log_prior'] = self.model['log_prior'](x)[0]
        self._state['grad_log_prior'] = self.model['log_prior'].grad(x)[0].flatten()

    def __getstate__(self):
        return self._state

    def __setstate__(self, state):
        self._state = state

    @property
    def log_likelihood(self):
        return self._state['log_likelihood']

    @property
    def log_prior(self):
        return self._state['log_prior']

    @property
    def num_params(self):
        return self._state['params'].shape[0]

    @property
    def params(self):
        return self._state['params']

    @params.setter
    def params(self, value):
        self._state['params'] = value
        self._eval_state()

    @property
    def grad_log_likelihood(self):
        return self._state['grad_log_likelihood']

    @property
    def grad_log_prior(self):
        return self._state['grad_log_prior']


def main(options):
    """
    The main function.
    """
    model = NativeModel(initialize_native_model(options.model))
    proposal = pm.MALAProposal(dt=options.dt)
    db_filename = options.db_filename
    if db_filename is None:
        db_filename = os.path.splitext(os.path.abspath(options.model))[0] + '_mcmc.h5' 
    mcmc = pm.MetropolisHastings(model,
                                 proposal=proposal,
                                 db_filename=db_filename)
    mcmc.sample(options.num_sample,
                num_thin=options.num_thin,
                num_burn=options.num_burn,
                verbose=True,
                stop_tuning_after=0)
    

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-m', '--model', metavar='FILE', dest='model',
                      help='specify the file containing the native model')
    parser.add_option('--dt', type=float, dest='dt', default=1.,
                      help='the proposal size')
    parser.add_option('--db-filename', metavar='FILE', dest='db_filename',
                      help='specify the file containing the output database')
    parser.add_option('--num-sample', type=int, dest='num_sample',
                      default=1000, help='the number of total samples to take')
    parser.add_option('--num-thin', type=int, dest='num_thin', default=100, 
                      help='the number of samples to drop between stores')
    parser.add_option('--num-burn', type=int, dest='num_burn', default=100,
                      help='the number of samples to be burned')
    options, args = parser.parse_args()
    main(options)
