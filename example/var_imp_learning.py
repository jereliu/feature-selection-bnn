"""Playground.

Note:
To inspect graph, run below command in the terminal
    > source activate tensorflow_gpuenv
    > tensorboard --logdir="./tmp"

then in browser, go to:
    http://jeremiah-predator:6006/

"""
from importlib import reload

import numpy as np

import tensorflow_probability as tfp

import inference.mcmc as mcmc

import util.data as data_util
import util.dtype as dtype_util
import util.visual as visual_util
import util.experiment as exp_util

tfd = tfp.distributions

WEIGHT_PRIOR_SD = np.sqrt(.1).astype(dtype_util.NP_DTYPE)

# if __name__ == "__main__":
logdir = "./tmp/"

data_config = {"n": 500, "d": 50, "d_true": 5,
               "data_gen_func": data_util.data_gen_func_linear,
               "random_seed_f": None}

model_config = {"n_node": 50, "n_layer": 2,
                "hidden_weight_sd": WEIGHT_PRIOR_SD,
                "output_weight_sd": .1}

mcmc_config = {"num_sample": int(5e3),
               "num_burnin": int(1e4),
               "num_pred_sample": 250}

# generate training data
(y_train, X_train,
 f_test, X_test, true_var_imp) = data_util.generate_data(**data_config)

# 1. Build Model and Inference Graph ################
(param_samples, is_accepted, param_names,
 model_fn, mcmc_graph) = exp_util.make_bnn_graph(X_train, y_train,
                                                 **mcmc_config,
                                                 **model_config)

# 2. Execute Training then Predict ################
param_sample_dict = mcmc.sample_parameter(param_samples, is_accepted,
                                          param_names, mcmc_graph)

(pred_sample, imp_sample,
 bias_samples) = mcmc.sample_predictive(mcmc_config["num_pred_sample"],
                                        param_sample_dict,
                                        model_fn, X_test)

# 4. Evaluate Model Fit ################
pred_mean = np.mean(pred_sample, 0)
var_imp_mean = np.mean(imp_sample, 0)

print("predict MSE:{:4f}".format(np.mean((pred_mean - f_test) ** 2)))
print("var_imp MSE:{:4f}".format(np.mean((var_imp_mean - true_var_imp) ** 2)))

# optionally, visualize
visual_util.plot_var_imp(imp_sample, true_var_imp, n_variable=50)
visual_util.plot_prediction(pred_sample, f_test)