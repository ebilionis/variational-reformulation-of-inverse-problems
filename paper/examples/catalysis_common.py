"""
Things that are common to the catalysis problem (no matter how you solve it).

Author:
    Ilias Bilionis

Date:
    9/9/2014

"""


from examples_common import *
import numpy as np


data = os.path.join(example_dir, 'data.txt')


def load_catalysis_data():
    """
    Loads the catalysis data and casts them to the right format.
    """
    data = np.loadtxt(os.path.join(example_dir, 'data.txt')).reshape((7, 6))
    y = data[:, 1:]
    return y.reshape((1, y.shape[0] * y.shape[1]))


def load_dimensionless_catalysis_data():
    """
    Loads a dimensionless version of the catalysis data.
    """
    return load_catalysis_data() / 500.


def plot_catalysis_output(fig_name,
                          Y,
                          y=load_dimensionless_catalysis_data(),
                          t=np.array([0.0, 1./6, 1./3, 1./2, 2./3, 5./6, 1.]),
                          colors=['b', 'r', 'g', 'k', 'm'],
                          linestyles=['', '--', '-.', '--+', ':'],
                          markerstyles=['o', 'v', 's', 'D', 'p'],
                          legend=False,
                          title=None):
    """
    Draw the output of the catalysis problem.
    
    :param fig_name:    A name for the figure.
    :param Y:           The samples observed.
    :param y:           The observations.
    :parma t:           The times of observations.
    """
    shape = (t.shape[0], Y.shape[1] / t.shape[0])
    y = y.reshape(shape)
    Y_m = np.percentile(Y, 50, axis=0).reshape(shape)
    Y_p05 = np.percentile(Y, 5, axis=0).reshape(shape)
    Y_p95 = np.percentile(Y, 95, axis=0).reshape(shape)
    Y_p05[Y_p05 <= 0.] = 0.
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in xrange(5):
        ax.plot(t, Y_m[:, i], colors[i] + linestyles[i], linewidth=2, markersize=5.)
        ax.fill_between(t, Y_p05[:, i], Y_p95[:, i], color=colors[i], alpha=0.5)
    for i in xrange(5):
        ax.plot(t, y[:, i], colors[i] + markerstyles[i], markersize=10)
    ax.set_xlabel('Time ($\\tau$)', fontsize=26)
    ax.set_ylabel('Concentration', fontsize=26)
    ax.set_title(title, fontsize=26)
    plt.setp(ax.get_xticklabels(), fontsize=26)
    plt.setp(ax.get_yticklabels(), fontsize=26)
    if legend:
        leg = plt.legend(['$\operatorname{NO}_3^-$', '$\operatorname{NO}_2^-$',
                          '$\operatorname{N}_2$', '$\operatorname{N}_2\operatorname{O}$',
                          '$\operatorname{NH}_3$'], loc='best')
        plt.setp(leg.get_texts(), fontsize=26)
    ax.set_ylim([0., 1.5])
    plt.tight_layout()
    print 'Writing:', fig_name
    plt.savefig(fig_name)
