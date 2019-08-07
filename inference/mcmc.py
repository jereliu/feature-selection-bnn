"""Functions to execute MCMC training for BNN."""
import time

import tqdm

import numpy as np

import tensorflow as tf
import tensorflow_probability.python.edward2 as ed

import util.bnn as model_util
import tensorflow_probability as tfp


def sample_parameter(sample_op, is_accepted_op, rv_names, mcmc_graph):
    """Performs MCMC sampling in session.

    Args:
        sample_op: (List of tf.Tensor) A list of Tensor of mcmc samples.
            shape (n_sample, variable_dim)
        is_accepted_op: (tf.Tensor) A vector indicating whether sample is accepted.
        mcmc_graph: (tf.Graph) A TF graph containing the model and sampling ops.
        rv_names: (List) A list of variable names sampled.

    Returns:
        (Dict of np.ndarray) A dictionary containing sampled values.
    """
    mcmc_sess = tf.Session(graph=mcmc_graph)

    with mcmc_sess.as_default():
        time_start = time.time()

        (samples_val, is_accepted_val) = mcmc_sess.run([sample_op, is_accepted_op])

        total_min = (time.time() - time_start) / 60.
        print('Acceptance Rate: {}'.format(np.mean(is_accepted_val)))
        print('Total time: {:.2f} min'.format(total_min))

    return dict(zip(rv_names, samples_val))


def define_mcmc(model_fn, X, y, num_sample, num_burnin):
    """Prepares Model Likelihood and defines MCMC sampling op.

    Args:
        model_fn: (function) A function of model definition.
        X: (Tensor or ndarray) A Tensor of input variables
        y: (Tensor or ndarray) A Tensor of response variables
        num_sample: (int) number of MCMC samples.
        num_burnin: (int) number of burn-in samples.

    Returns:
        samples: (List of tf.Tensor) A list of Tensor of mcmc samples.
            shape (n_sample, variable_dim)
        is_accepted: (tf.Tensor) A vector indicating whether sample is accepted.
    """
    # define model likelihood and initial states
    log_prob_fn = model_util.make_log_prob_fn(model_fn, X, y)
    init_state, rv_names = _make_init_state(model_fn, X)

    # define mcmc kernel and sampler
    samples, is_accepted = _sample_hmc_chain(log_prob_fn, init_state,
                                             num_sample, num_burnin)
    return samples, is_accepted, rv_names


def sample_predictive(n_sample, param_sample_dict, model_fn, X, sample_bias=False):
    """Produce samples from the posterior distribution.

    Args:
        n_sample: (int) Number of samples to obtain from the predictive posterior.
        param_sample_dict: (Dict of np.ndarray) A dictionary containing sampled values.
        model_fn: (function) A function of model definition.
        X_pred: (Tensor or ndarray) A Tensor of input variables to evaluate prediction at.
        sample_bias: (bool) Whether to sample bias.

    Returns:
        pred_sample: (np.ndarray) Samples for y prediction given X_pred.
        imp_sample: (np.ndarray) Samples for un-centered variable importance.
        imp_bias_sample: (np.ndarray or None)
            Samples for variable importance bias if sample_bias=True. Otherwise None.
    """
    # TODO: (jereliu) convert sample to correlated Empirical RVs

    n_param_sample = list(param_sample_dict.values())[0].shape[0]
    pred_graph = tf.Graph()

    # define sampling graph
    with pred_graph.as_default():
        # collect posterior sample for variable importance and prediction
        post_pred_samples = []
        var_imp_samples = []
        var_imp_bias_samples = []

        # iterate through parameter samples,
        # generate one predictive sample for each param sample
        for sample_id in tqdm.tqdm(range(n_param_sample - n_sample, n_param_sample)):
            sample_pred_dict = {rv_name: rv_sample[sample_id]
                                for rv_name, rv_sample in param_sample_dict.items()}
            with ed.interception(ed.make_value_setter(**sample_pred_dict)):
                _, network_pred, var_imp, var_imp_bias, _, _ = model_fn(X)

            post_pred_samples.append(network_pred)
            var_imp_samples.append(var_imp)
            var_imp_bias_samples.append(var_imp_bias)

        pred_graph.finalize()

    # evaluate sampling graph in session
    with tf.Session(graph=pred_graph) as sess:
        pred_sample, imp_sample = sess.run([post_pred_samples, var_imp_samples, ])

        pred_sample = np.asarray(pred_sample)
        imp_sample = np.asarray(imp_sample)
        bias_sample = None

        if sample_bias:
            bias_sample = sess.run(var_imp_bias_samples)
            bias_sample = np.asarray(bias_sample)

    return pred_sample, imp_sample, bias_sample


def _sample_hmc_chain(log_prob_fn, init_state,
                      num_sample, num_burnin, hmc_kernel=None):
    """Performs MCMC sampling using HMC kernel."""
    # by default use adaptive hmc kernel
    if not hmc_kernel:
        hmc_kernel = _make_adaptive_hmc_kernel(log_prob_fn, num_burnin)

    samples, is_accepted = tfp.mcmc.sample_chain(
        num_results=num_sample,
        num_burnin_steps=num_burnin,
        current_state=init_state,
        kernel=hmc_kernel,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

    return samples, is_accepted


def _make_adaptive_hmc_kernel(log_prob_fn, num_burnin):
    """Defines an HMC kernel using adaptive step sizes."""
    hmc_base_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_fn,
        step_size=0.01,
        num_leapfrog_steps=5)

    hmc_adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=hmc_base_kernel,
        num_adaptation_steps=int(num_burnin * 0.8))

    return hmc_adaptive_kernel


def _make_init_state(model_fn, X):
    """Makes a list of initial state for MCMC training.
    Args:
        model_fn: (function) A model function.
        X: (Tensor or ndarray) A Tensor of input variables

    Returns:
        init_state (list of tf.Tensors) List of sampled initial
            values for random variables in the model.
        rv_names (List of str) List of random variable names in
            the state
    """
    rv_dict = model_util.get_variable_dict(model_fn, X)
    rv_names = list(rv_dict.keys())

    # produce initial state but exclude response
    init_state = [
        rv_value.distribution.sample() for
        rv_name, rv_value in rv_dict.items() if rv_name != 'y'
    ]

    return init_state, rv_names
