"""Utility function for generating plots."""
import os
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def plot_var_imp(imp_sample, true_var_imp, n_variable=50):
    """Plots violin plots of variable importance v.s. truth."""
    # produce pandas data frame for plotting
    n_sample, _ = imp_sample.shape

    feature_names = ["x_{}".format(p) for p in range(n_variable)]
    var_imp_data = pd.DataFrame(
        {"feature": np.tile(feature_names, n_sample),
         "importance": np.hstack(imp_sample[:, :n_variable])})

    sns.violinplot(x="feature", y="importance", data=var_imp_data)
    plt.scatter(x=range(n_variable),
                y=true_var_imp[:n_variable], c="red")


def plot_prediction(pred_sample, true_func, save_addr="./tmp/"):
    """Plots model prediction v.s. truth."""
    pred_mean = np.mean(pred_sample, 0)

    plt.scatter(pred_mean, true_func)
    plt.plot(np.arange(np.min(true_func), np.max(true_func), 0.1),
             np.arange(np.min(true_func), np.max(true_func), 0.1), c='orange')
