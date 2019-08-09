"""Utility functions for simulation experiments."""
import datetime
import tensorflow as tf

import model.bnn as model
import inference.mcmc as mcmc


def make_bnn_graph(X, y, num_sample, num_burnin, **bnn_kwargs):
    """Makes a TF Graph containing model and inference nodes.

    Args:
        X: (Tensor or ndarray) A Tensor of input variables
        y: (Tensor or ndarray) A Tensor of response variables
        num_sample: (int) Number of MCMC samples for neural network parameters.
        num_burnin: (int) Number of burn-in samples for MCMC chain.
        **bnn_kwargs: (Dict) Keyword arguments for model.define_bnn()

    Returns:
        param_samples (List of tf.Tensor) A list of Tensor of mcmc samples.
            shape (n_sample, variable_dim).
        is_accepted: (tf.Tensor) A vector indicating whether sample is accepted.
        param_names (List of str) List of random variable names in the state.
        model_fn: (function) A function of model definition.
        bnn_graph (tf.Graph) A TF Graph containing model and inference nodes.
    """
    bnn_graph = tf.Graph()

    with bnn_graph.as_default():
        # 1. Model Definition ################
        # define model
        model_fn = model.define_bnn(**bnn_kwargs)

        # 2. Training Definition ################
        param_samples, is_accepted, param_names = (
            mcmc.define_mcmc(model_fn, X, y, num_sample, num_burnin))

        bnn_graph.finalize()

    return param_samples, is_accepted, param_names, model_fn, bnn_graph


def make_stamps(data_config, model_config):
    time_stamp = datetime.datetime.now().strftime("%h%d_%H%M%S")
    config_stamp = "n{}d{}_{}_l{}k{}".format(data_config['n'], data_config['d'],
                                             data_config['data_type'],
                                             model_config['n_node'],
                                             model_config['n_layer'])
    config_detail = \
        "n={}, d={}, {}, L={}, K={}".format(data_config['n'], data_config['d'],
                                            data_config['data_type'],
                                            model_config['n_node'],
                                            model_config['n_layer'])
    return time_stamp, config_stamp, config_detail
