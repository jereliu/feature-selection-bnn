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

import tensorflow as tf
import tensorflow_probability as tfp

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

if __name__ == "__main__":
    import pandas as pd
    from util.config import data_config, model_config, mcmc_config

    record_addr = os.path.join(DEFAULT_LOG_DIR, "var_imp_learning.csv")

    if os.path.isfile(record_addr):
        exp_records = pd.read_csv(record_addr)
    else:
        exp_records = pd.DataFrame(columns=["n", "d", "f", "l", "k",
                                            "pred_mse", "var_imp_mse"])

    data_dimn_list = [35, 50, 100, 150, 175, 200]
    data_size_list = [
        # 100, 125, 150, 200, 250, 300,
        # 400, 500, 750, 1000, 1500,
        2000, 5000, 7500, 9000]
    data_type_list = ["sobolev", "barron", "linear"]

    for rep in range(2, EXPERIMENT_REPEAT):
        for d_type in data_type_list:
            data_config["data_type"] = d_type
            for d in data_dimn_list:
                data_config["d"] = d
                for n in data_size_list:
                    data_config["n"] = n
                    # execute experiment
                    print("===========================================")
                    pred_mse, var_imp_mse, _, _, _ = \
                        exp_util.run_experiment(data_config,
                                                model_config,
                                                mcmc_config)
                    print("===========================================")

                    # record result and save
                    new_record = {"n": data_config["n"],
                                  "d": data_config["d"],
                                  "f": data_config["data_type"],
                                  "l": model_config["n_layer"],
                                  "k": model_config["n_node"],
                                  "pred_mse": pred_mse,
                                  "var_imp_mse": var_imp_mse}
                    exp_records.loc[len(exp_records)] = new_record

                    exp_records.to_csv(record_addr, index=False)
                    exp_records.to_csv("/home/jeremiah/Dropbox/exp_res.csv",
                                       index=False)

    # quick visual of var importance MSE by dimension
    for d_type in data_type_list:
        exp_records_plot = exp_records[exp_records.f == d_type]
        exp_records_plot = exp_records_plot.sort_values(by='d')
        exp_records_plot = exp_records_plot[exp_records_plot.d != 25]

        plt.figure(figsize=(10, 8))
        for dim_value in exp_records_plot["d"].unique():
            dim_data = exp_records_plot[exp_records_plot.d == dim_value]
            dim_data_mean = dim_data.groupby('n').median()
            plt.plot(dim_data_mean.var_imp_mse)

        plt.legend(exp_records_plot["d"].unique(), loc='upper right')
        plt.title("Variable Importance MSE v.s. Sample Size", fontsize=16)
        plt.xlabel('n', fontsize=16)
        plt.ylabel('Variable Importance MSE', fontsize=16)
        # plt.ylim((0, 5))
        plt.savefig("learning_var_imp_{}.png".format(d_type))
        plt.close()

        plt.figure(figsize=(10, 8))
        for dim_value in exp_records_plot["d"].unique():
            dim_data = exp_records_plot[exp_records_plot.d == dim_value]
            dim_data_mean = dim_data.groupby('n').median()
            plt.plot(dim_data_mean.pred_mse)

        plt.legend(exp_records_plot["d"].unique(), loc='upper right')
        plt.title("Prediction MSE v.s. Sample Size", fontsize=16)
        plt.xlabel('n', fontsize=16)
        plt.ylabel('Prediction MSE', fontsize=16)
        # plt.ylim((0, 5))
        plt.savefig("learning_pred_{}.png".format(d_type))
        plt.close()
