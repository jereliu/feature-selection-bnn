"""Playground.

Note:
To inspect graph, run below command in the terminal
    > source activate tensorflow_gpuenv
    > tensorboard --logdir="./tmp"

then in browser, go to:
    http://jeremiah-predator:6006/

"""
import os
from importlib import reload

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import inference.mcmc as mcmc

import util.data as data_util
import util.visual as visual_util
import util.experiment as exp_util

import matplotlib.pyplot as plt
import seaborn as sns

tfd = tfp.distributions

#
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

#
DEFAULT_LOG_DIR = "./experiment/"
EXPERIMENT_REPEAT = 5

os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)


def plot_and_save(plot_func, logdir,
                  mse_val, data_config, model_config,
                  figsize=(8, 6), **plot_kwargs):
    time_stamp, config_stamp, config_detail = exp_util.make_stamps(data_config, model_config)
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


def run_experiment(data_config, model_config, mcmc_config, logdir=DEFAULT_LOG_DIR):
    """Execute an experiment.

    Args:
        data_config: (Dict) Dictionary of Data configs.
        model_config: (Dict) Dictionary of Model configs.
        mcmc_config: (Dict) Dictionary of MCMC configs.
        logdir: (str) File address to save result to

    Returns:
        pred_mse, var_imp_mse (float)
            Mean Squared Error for prediction and variable importance
            estimation.
    """
    # generate training data
    print("Data: n={}, p={}".format(data_config["n"], data_config["d"]))
    (y_train, X_train,
     f_test, X_test, true_var_imp) = data_util.generate_data(**data_config)

    # 1. Build Model and Inference Graph ################
    print("Model: l={}, k={}".format(model_config["n_node"], model_config["n_layer"]))
    (param_samples, is_accepted, param_names,
     model_fn, mcmc_graph) = exp_util.make_bnn_graph(X_train, y_train,
                                                     **mcmc_config,
                                                     **model_config)

    # 2. Execute Training then Predict ################
    print("Executing Parameter Sampling...")
    param_sample_dict = mcmc.sample_parameter(param_samples, is_accepted,
                                              param_names, mcmc_graph)

    print("Executing Prediction...")
    (pred_sample, imp_sample,
     bias_samples) = mcmc.sample_predictive(mcmc_config["num_pred_sample"],
                                            param_sample_dict,
                                            model_fn, X_test)

    # 4. Evaluation Metric ################
    pred_mean = np.mean(pred_sample, 0)
    var_imp_mean = np.mean(imp_sample, 0)

    pred_mse = np.mean((pred_mean - f_test) ** 2)
    var_imp_mse = np.mean((var_imp_mean - true_var_imp) ** 2)
    print("VariableImp MSE: {:4f}".format(var_imp_mse))
    print("Prediction MSE: {:4f}".format(pred_mse))

    # 4. Model Fit Visual ################
    plot_and_save(visual_util.plot_var_imp, logdir,
                  var_imp_mse, data_config, model_config,
                  figsize=(8, 6),
                  imp_sample=imp_sample,
                  true_var_imp=true_var_imp,
                  n_variable=np.min((50, data_config["d"])))

    return pred_mse, var_imp_mse


if __name__ == "__main__":
    import pandas as pd
    from util.config import data_config, model_config, mcmc_config

    record_addr = os.path.join(DEFAULT_LOG_DIR, "var_imp_learning.csv")

    if os.path.isfile(record_addr):
        exp_records = pd.read_csv(record_addr)
    else:
        exp_records = pd.DataFrame(columns=["n", "d", "f", "l", "k",
                                            "pred_mse", "var_imp_mse"])

    data_dimn_list = [25, 50, 100, 150, 175, 200]
    data_size_list = [100, 125, 150, 200, 250, 300, 400, 500, 750, 1000, 1500]
    data_type_list = ["linear"]

    for f_name in data_type_list:
        data_config["data_gen_func"] = f_name
        for d in data_dimn_list:
            data_config["d"] = d
            for n in data_size_list:
                data_config["n"] = n
                for rep in range(EXPERIMENT_REPEAT):
                    # execute experiment
                    print("===========================================")
                    pred_mse, var_imp_mse = run_experiment(data_config,
                                                           model_config,
                                                           mcmc_config)
                    print("===========================================")

                    # record result and save
                    new_record = {"n": data_config["n"],
                                  "d": data_config["d"],
                                  "f": data_config["data_gen_func"],
                                  "l": model_config["n_layer"],
                                  "k": model_config["n_node"],
                                  "pred_mse": pred_mse,
                                  "var_imp_mse": var_imp_mse}
                    exp_records.loc[len(exp_records)] = new_record

                    exp_records.to_csv(record_addr, index=False)
                    exp_records.to_csv("/home/jeremiah/Dropbox/exp_res.csv",
                                       index=False)

    # quick visual of var importance MSE by dimension
    for dim_value in exp_records["d"].unique():
        dim_data = exp_records[exp_records.d == dim_value]
        dim_data_mean = dim_data.groupby('n').median()
        plt.plot(dim_data_mean.n, dim_data_mean.var_imp_mse)

    plt.legend(exp_records["d"].unique(), loc='upper right')
    plt.title("Variable Importance MSE v.s. Sample Size")
    plt.ylim((0, 5))
    plt.savefig("learning_var_imp.png")
    plt.close()

    for dim_value in exp_records["d"].unique():
        dim_data = exp_records[exp_records.d == dim_value]
        dim_data_mean = dim_data.groupby('n').median()
        plt.plot(dim_data_mean.n, dim_data_mean.pred_mse)

    plt.legend(exp_records["d"].unique(), loc='upper right')
    plt.title("Prediction MSE v.s. Sample Size")
    plt.ylim((0, 5))
    plt.savefig("learning_pred.png")
    plt.close()
