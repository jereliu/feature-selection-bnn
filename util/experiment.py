"""Utility functions for simulation experiments."""
import os
import datetime

import numpy as np

import tensorflow as tf

import model.bnn as model
import inference.mcmc as mcmc

import util.data as data_util
import util.visual as visual_util

import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_LOG_DIR = "./experiment/"


def run_experiment(data_config, model_config, mcmc_config,
                   sample_bias=False, logdir=DEFAULT_LOG_DIR):
    """Execute an experiment.

    Args:
        data_config: (Dict) Dictionary of Data configs.
        model_config: (Dict) Dictionary of Model configs.
        mcmc_config: (Dict) Dictionary of MCMC configs.
        sample_bias: (bool) Whether to sample bias terms.
            If so then also compute quantiles for UQ-based variable selection.
        logdir: (str) File address to save result to

    Returns:
        pred_mse, var_imp_mse (float)
            Mean Squared Error for prediction and variable importance
            estimation.
    """
    # generate training data
    print("Data: n={}, d={}, d_true={}, f={}".format(
        data_config["n"], data_config["d"],
        data_config["d_true"], data_config["data_type"]))
    (y_train, X_train, f_train,
     f_test, X_test, true_var_imp) = data_util.generate_data(**data_config)

    # 1. Build Model and Inference Graph ################
    print("Model: k={}, l={}, sd=({:.3f}, {:.3f})".format(
        model_config["n_node"], model_config["n_layer"],
        model_config["hidden_weight_sd"], model_config["output_weight_sd"]))
    (param_samples, is_accepted, param_names,
     model_fn, mcmc_graph) = make_bnn_graph(X_train, y_train,
                                            **mcmc_config,
                                            **model_config)

    # 2. Execute Training then Predict ################
    print("Executing Parameter Sampling...")
    param_sample_dict = mcmc.sample_parameter(param_samples, is_accepted,
                                              param_names, mcmc_graph)

    print("Executing Prediction...")
    (pred_sample, imp_sample,
     bias_sample) = mcmc.sample_predictive(mcmc_config["num_pred_batch"],
                                           param_sample_dict,
                                           model_fn, X_test,
                                           sample_bias=sample_bias)
    selected_set = set()
    if sample_bias:
        imp_sample_centered = imp_sample - bias_sample
        # compute confidence threshold
        q = 0.1
        q_marg = np.quantile(imp_sample_centered, q=q/2, axis=0)
        selected_set = set(np.where(q_marg > 0)[0])

    # 4. Evaluation Metric ################
    # 4.1. learning variable importance
    pred_mean = np.mean(pred_sample, 0)
    var_imp_mean = np.mean(imp_sample, 0)

    pred_mse = np.mean((pred_mean - f_test) ** 2 / np.var(f_test))
    var_imp_mse = np.mean((var_imp_mean - true_var_imp) ** 2)
    print("Prediction MSE: {:4f}".format(pred_mse))
    print("VariableImp MSE: {:4f}".format(var_imp_mse))

    # 4.2 variable selection
    precision = None
    recall = None
    specificity = None

    if sample_bias:
        true_pos_set = set(range(data_config['d_true']))
        true_neg_set = set(range(data_config['d_true'], data_config['d']))

        tp_set = selected_set.intersection(true_pos_set)
        tn_set = true_neg_set.difference(selected_set)
        fn_set = true_pos_set.difference(selected_set)
        fp_set = selected_set.difference(true_pos_set)

        if (len(tp_set) + len(fp_set))>0:
            precision = len(tp_set)/(len(tp_set) + len(fp_set))
        else:
            precision = 0.

        if (len(tp_set) + len(fn_set))>0:
            recall = len(tp_set)/(len(tp_set) + len(fn_set))
        else:
            recall = 0.

        if (len(tn_set) + len(fp_set)) > 0:
            specificity = len(tn_set)/(len(tn_set) + len(fp_set))
        else:
            specificity = 0.

        if precision > 0 and recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        print("Precision: {:4f}".format(precision))
        print("Recall: {:4f}".format(recall))
        print("F1: {:4f}".format(f1))

    # 4. Model Fit Visual ################
    plot_and_save(visual_util.plot_var_imp, logdir,
                  var_imp_mse, data_config, model_config,
                  figsize=(8, 6),
                  imp_sample=imp_sample,
                  true_var_imp=true_var_imp,
                  n_variable=np.min((50, data_config["d"])))

    return (pred_mse, var_imp_mse,
            precision, recall, specificity, f1,
            selected_set, imp_sample_centered)


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


def plot_and_save(plot_func, logdir,
                  mse_val, data_config, model_config,
                  figsize=(8, 6), **plot_kwargs):
    time_stamp, config_stamp, config_detail = make_stamps(data_config, model_config)
    save_addr = os.path.join(os.path.abspath(logdir), config_stamp)
    file_addr = "{}/{}_{}_mse_{:3f}.png".format(save_addr, config_stamp, time_stamp, mse_val)

    os.makedirs(save_addr, exist_ok=True)
    plt.ioff()

    plt.figure(figsize=figsize)
    plot_func(**plot_kwargs)
    plt.title("{}, MSE={:5f}".format(config_detail, mse_val))
    plt.savefig(file_addr)
    plt.close()

    plt.ion()
