from time import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg as linalg

from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.datasets.samples_generator import make_blobs, make_spd_matrix

colors = ['c', 'm', 'y', 'k',
          'r', 'g', 'b', 'y',
          'navy', 'turquoise', 'darkorange']


def covariances_(estimator):
    if estimator.precision_type is 'full':
        return [linalg.inv(prec) for prec in estimator.precisions_]
    elif estimator.precision_type is 'tied':
        return linalg.inv(estimator.precisions_)
    else:
        return 1. / estimator.precisions_


def generate_data(n_samples, means, covars):
    n_components = len(n_samples)
    X = np.vstack([rng.multivariate_normal(means[j], covars[j], n_samples[j])
                  for j in range(n_components)])
    y = np.concatenate([j * np.ones(n_samples[j])
                       for j in range(n_components)])
    return X, y


def plot_ellipses(means, covars, matrix_type, ax):
    for n in range(means.shape[0]):
        if matrix_type == 'full':
            cov = covars[n][:2, :2]
        elif matrix_type == 'tied':
            cov = covars[:2, :2]
        elif matrix_type == 'diag':
            cov = np.diag(covars[n][:2])
        else:
            cov = np.eye(4) * covars[n]
        v, w = np.linalg.eigh(cov)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2 * np.sqrt(2) * np.sqrt(v)
        if(means.shape[0] > len(colors)):
            ell = mpl.patches.Ellipse(means[n, :2], v[0], v[1], 180 + angle)
        else:
            ell = mpl.patches.Ellipse(means[n, :2], v[0], v[1], 180 + angle,
                                      color=colors[n])
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


def plot_data(X, y, estimator):
    plt.clf()
    h = plt.subplot(111)
    plt.axis('equal')
    for n, color in enumerate(range(n_components)):
        data = X[y == n]

        plt.scatter(data[:, 0], data[:, 1], s=0.8)
        plot_ellipses(estimator.means_, covariances_(estimator),
                      estimator.covariance_type if isinstance(estimator, GaussianMixturePrecision) else
                      estimator.precision_type, h)
    plt.draw()


def plot(fit, predict, gmm_error1, gmm_error2, sizes, xlabel, gmm_class_name1, gmm_class_name2):
    """Plot the results."""

    idx = np.arange(fit.shape[1])

    plt.figure( figsize=(14, 4))
    plt.plot( fit.mean(axis=0), c='b', label="Fitting")
    plt.plot( predict.mean(axis=0), c='r', label="Prediction")
    plt.plot( [0, fit.shape[1]], [1, 1], c='k', label="Baseline" )

    plt.fill_between( idx, fit.min(axis=0), fit.max(axis=0), color='b', alpha=0.3 )
    plt.fill_between( idx, predict.min(axis=0), predict.max(axis=0), color='r', alpha=0.3 )

    plt.xticks(idx, sizes, rotation=65, fontsize=14)
    plt.xlabel('{}'.format(xlabel), fontsize=14)
    plt.ylabel('%s is x times faster than %s' %(gmm_class_name2, gmm_class_name1), fontsize=14)
    plt.legend(fontsize=12, loc=4)
    plt.show()


    plt.figure( figsize=(14, 4))
    plt.plot( 1 - gmm_error1.mean(axis=0), alpha=0.5, c='b', label="%s accuracy" % gmm_class_name2)
    plt.plot( 1 - gmm_error2.mean(axis=0), alpha=0.5, c='r', label="%s accuracy" % gmm_class_name1)

    plt.fill_between( idx, 1-gmm_error1.min(axis=0), 1-gmm_error1.max(axis=0), color='b', alpha=0.3 )
    plt.fill_between( idx, 1-gmm_error2.min(axis=0), 1-gmm_error2.max(axis=0), color='r', alpha=0.3 )

    plt.xticks( idx, sizes, rotation=65, fontsize=14)
    plt.xlabel( '{}'.format(xlabel), fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=14)
    plt.show()