import numpy as np

import util.dtype as dtype_util
import util.data as data_util

# default configurations (sqrt(.1) for linear, sqrt(.1) for barron)
WEIGHT_PRIOR_SD = np.sqrt(.1).astype(dtype_util.NP_DTYPE)

data_config = {"n": 500, "d": 200, "d_true": 5,
               "data_type": "sobolev",
               "random_seed_x": 100,
               "random_seed_f": 500}

model_config = {"n_node": 50, "n_layer": 2,
                "hidden_weight_sd": WEIGHT_PRIOR_SD,
                "output_weight_sd": .1}

mcmc_config = {"num_sample": int(1e3),
               "num_burnin": int(2e3),
               "num_pred_batch": 100,
               "num_pred_epoch": 3}
