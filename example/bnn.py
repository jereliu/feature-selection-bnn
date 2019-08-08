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
import pandas as pd

import tensorflow_probability as tfp

import inference.mcmc as mcmc

import util.data as data_util
import util.dtype as dtype_util
import util.experiment as exp_util

import seaborn as sns
import matplotlib.pyplot as plt

tfd = tfp.distributions

WEIGHT_PRIOR_SD = np.sqrt(.1).astype(dtype_util.NP_DTYPE)

# if __name__ == "__main__":
logdir = "./tmp/"

n_train = 1000
n_feature = 50
n_feature_true = 5

num_sample = int(5e3)
num_burnin = int(1e4)
num_pred_sample = 250

model_config = {"n_node": 50, "n_layer": 2,
                "hidden_weight_sd": WEIGHT_PRIOR_SD,
                "output_weight_sd": .1}

# generate training data
(y_train, X_train,
 f_train, true_var_imp) = data_util.generate_data(n=n_train,
                                                  d=n_feature,
                                                  d_true=n_feature_true,
                                                  data_gen_func=data_util.data_gen_func_linear,
                                                  random_seed_f=50)

# 1. Build Model and Inference Graph ################
(param_samples, is_accepted, param_names,
 model_fn, mcmc_graph) = exp_util.make_bnn_graph(X_train, y_train,
                                                 num_sample, num_burnin,
                                                 **model_config)

# 2. Execute Training then Predict ################
param_sample_dict = mcmc.sample_parameter(param_samples, is_accepted,
                                          param_names, mcmc_graph)

(pred_sample, imp_sample,
 bias_samples) = mcmc.sample_predictive(num_pred_sample,
                                        param_sample_dict,
                                        model_fn, X_train)

# 4. Evaluate Model Fit ################
# check in-sample predictive accuracy
posterior_mean = np.mean(pred_sample, 0)
plt.scatter(posterior_mean, f_train)
plt.plot(np.arange(np.min(f_train), np.max(f_train), 0.1),
         np.arange(np.min(f_train), np.max(f_train), 0.1), c='orange')

# produce pandas data frame for plotting
n_feature_plot = 50  # n_feature
feature_names = ["x_{}".format(p) for p in range(n_feature_plot)]
var_imp_data = pd.DataFrame(
    {"feature": np.tile(feature_names, num_pred_sample),
     "importance": np.hstack(imp_sample[:, :n_feature_plot])})

sns.violinplot(x="feature", y="importance", data=var_imp_data)
plt.scatter(x=range(n_feature_plot),
            y=true_var_imp[:n_feature_plot], c="red")
