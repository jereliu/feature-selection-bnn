import numpy as np

import util.data as data_util
import util.experiment as exp_util

from util.config import data_config, model_config, mcmc_config

import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri

numpy2ri.activate()

# generate data
print("Data: n={}, d={}, d_true={}, f={}".format(
    data_config["n"], data_config["d"],
    data_config["d_true"], data_config["data_type"]))
(y_train, X_train, f_train,
 f_test, X_test, true_var_imp) = data_util.generate_data(**data_config)

np.savetxt("X_train.csv", X_train, delimiter=",")
np.savetxt("y_train.csv", y_train, delimiter=",")
