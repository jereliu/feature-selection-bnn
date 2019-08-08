"""Utility functions for generating data."""
import inspect

import numpy as np

import util.dtype as dtype_util


def data_gen_func_linear(X, d_true, snr=2, random_seed=100):
    """A Linear Function to generates toy data for training.

    Args:
        X: (np.ndarray) A matrix of input features
        d_true: (int)  Number of real input features
        snr: (float) Signal-to-noise ratio of the true function.
        random_seed: (int) Random seed to set for data generation.

    Returns:
        y: (np.ndarray of NP_DTYPE) A vector of response, shape (n, )
        variable_importance: (np.ndarray of NP_DTYPE)
            A vector of variable Importance for each input features, shape (d, )
    """
    np.random.seed(random_seed)

    n, d = X.shape

    X_true = X[:, :d_true]
    linear_coef = np.random.normal(loc=1., scale=.25, size=d_true)

    # produce function
    f = X_true.dot(linear_coef)

    snr_adjust = snr / np.std(f)
    f = snr_adjust * f
    y = f + np.random.normal(0., 1., size=n)

    # produce variable importance
    var_imp = np.zeros(shape=(d,))
    var_imp[:d_true] = (snr_adjust * linear_coef) ** 2

    return y, f, var_imp


def generate_data(n=1000, n_test=1000,
                  d=10, d_true=5,
                  data_gen_func=data_gen_func_linear,
                  random_seed_x=None, random_seed_f=None):
    """Generates toy data for training.

    Args:
        n: (int) Number of observations
        n_test: (int) Number of testing observations.
        d: (int) Number of Input Features
        d_true: (int)  Number of Real Input Features.
        data_gen_func: (function) A function that takes in feature, d_true, and
            return  (y, variable_importance)
        random_seed_x: (int) Random seed for generating features.
        random_seed_y: (int) Random seed for generating response.

    Returns:
        y_train: (np.ndarray of NP_DTYPE) A vector of response, shape (n, )
        X_train: (np.ndarray of NP_DTYPE)  A matrix of input features between (0, 1),
            shape (n, d).
        f_test: (np.ndarray of NP_DTYPE) A vector of response, shape (n_test, )
        X_test: (np.ndarray of NP_DTYPE)  A matrix of input features between (0, 1),
            shape (n_test, d).

        variable_importance: (np.ndarray of NP_DTYPE)
            A vector of variable Importance for each input features, shape (d, )

    Raises:
        (ValueError): If d_true >= d
        (ValueError): If signature of data_gen_func doesn't contain 'X' and 'd_true'
    """

    if d_true >= d:
        raise ValueError("d_true cannot be greater than d.")
    for arg_name in ['X', 'd_true']:
        if not arg_name in inspect.getfullargspec(data_gen_func).args:
            raise ValueError("'data_gen_func' signature "
                             "should contain arg name '{}'".format(arg_name))

    # generate features
    np.random.seed(random_seed_x)
    X = np.random.uniform(0., 1., size=(n + n_test, d))

    # generate data, standardize, and convert data type
    y, f, var_imp = data_gen_func(X, d_true, random_seed=random_seed_f)
    f = f - np.mean(y)
    y = y - np.mean(y)

    y = y.astype(dtype_util.NP_DTYPE)
    X = X.astype(dtype_util.NP_DTYPE)

    X_train = X[:n, :]
    y_train = y[:n]

    X_test = X[n + 1:, :]
    f_test = f[(n + 1):]

    var_imp = var_imp.astype(dtype_util.NP_DTYPE)

    return y_train, X_train, f_test, X_test, var_imp
