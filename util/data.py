"""Utility functions for generating data."""
import inspect

import tensorflow as tf
from tensorflow_probability import positive_semidefinite_kernels as tfk

import numpy as np

import util.dtype as dtype_util

AVAIL_DATA_TYPE = ["linear", "barron", "sobolev"]


def generate_noiseless_data(X, d_true,
                            data_type,
                            random_seed=100):
    """A Linear Function to generates toy data for training.

    Args:
        X: (np.ndarray) A matrix of input features.
        d_true: (int)  Number of real input features.
        data_type: (str)  Types of data to generate.
        random_seed: (int) Random seed to set for data generation.

    Returns:
        y: (np.ndarray of NP_DTYPE) A vector of response, shape (n, ).
        variable_importance: (np.ndarray of NP_DTYPE) A vector of variable
         importance for each input features, shape (d, ).

    Raises:
        (ValueError) If data_type does not belong to AVAIL_DATA_TYPE.
    """
    if not data_type in AVAIL_DATA_TYPE:
        raise ValueError("data type '{}' not available.".format(data_type))

    np.random.seed(random_seed)

    n, d = X.shape
    x = tf.Variable(initial_value=X, dtype=dtype_util.TF_DTYPE)

    if data_type == "linear":
        # produce coefficient
        linear_coef = np.zeros(shape=d).astype(dtype_util.NP_DTYPE)
        linear_coef[:d_true] = np.random.normal(loc=1., scale=.25,
                                                size=d_true)
        # produce function
        f = tf.tensordot(x, linear_coef, axes=1)
    elif data_type == "barron":
        if not d_true == 5:
            raise ValueError("Barron class function only supports d_true=5.")

        f = (10 * tf.sin(tf.reduce_max(x[:, :2], axis=1)) +
             tf.reduce_max(x[:, 2:5]) ** 3) / (1 + (x[:, 0] + x[:, 4]) ** 2) + \
            tf.sin(0.5 * x[:, 2]) * (1 + tf.exp(x[:, 3] - 0.5 * x[:, 2])) + \
            x[:, 2] ** 2 + 2 * tf.sin(x[:, 3]) + 2 * x[:, 4]
    elif data_type == "sobolev":
        sobolev_kernel = tfk.MaternThreeHalves(amplitude=1, length_scale=.1)
        sobolev_mat = sobolev_kernel.matrix(x[:, :d_true], x[:, :d_true])

        alpha = np.random.normal(size=n).astype(dtype_util.NP_DTYPE)
        f = tf.tensordot(sobolev_mat, alpha, axes=1)

        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     sess.run(tf.local_variables_initializer())
        #     f_val, x_val = sess.run([f, x[:, :d_true]])
        #
        #     import pandas as pd
        #     pd_plot = pd.DataFrame({"x": x_val.flatten(), "f": f_val.flatten()})
        #     pd_plot = pd_plot.sort_values(by="x")
        #     plt.plot(pd_plot.x, pd_plot.f)
    else:
        raise ValueError("data type {} not available.".format(data_type))

    # produce variable importance
    var_imp = tf.reduce_mean(tf.gradients(f, x)[0] ** 2, axis=0)

    # evaluate
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        f_val, var_imp_val = sess.run([f, var_imp])

    return f_val, var_imp_val


def generate_data(n=1000, n_test=1000,
                  d=10, d_true=5, snr=2,
                  data_type="linear",
                  random_seed_x=None, random_seed_f=None, **kwargs):
    """Generates toy data for training.

    Args:
        n: (int) Number of observations
        n_test: (int) Number of testing observations.
        d: (int) Number of Input Features
        d_true: (int)  Number of Real Input Features.
        snr: (float) Signal-to-noise ratio of the true function.
        data_type: (function) A function that takes in feature, d_true, and
            return  (y, variable_importance)
        random_seed_x: (int) Random seed for generating features.
        random_seed_y: (int) Random seed for generating response.
        **kwargs: Additional keyword arguments.

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

    # generate features
    np.random.seed(random_seed_x)

    n_all = n + n_test
    X = np.random.uniform(0., 1., size=(n_all, d))

    # generate data, standardize, and convert data type
    f, var_imp = generate_noiseless_data(X, d_true,
                                         data_type=data_type,
                                         random_seed=random_seed_f)

    # standardization, adjust signal to noise ratio, produce y
    f = f - np.mean(f)
    snr_adjust = snr / np.std(f)
    f = snr_adjust * f
    var_imp = var_imp * (snr_adjust ** 2)

    y = f + np.random.normal(0., 1., size=n_all)

    # outcome formatting
    y = y.astype(dtype_util.NP_DTYPE)
    X = X.astype(dtype_util.NP_DTYPE)
    var_imp = var_imp.astype(dtype_util.NP_DTYPE)

    X_train = X[:n, :]
    y_train = y[:n]

    X_test = X[n + 1:, :]
    f_test = f[(n + 1):]

    return y_train, X_train, f_test, X_test, var_imp
