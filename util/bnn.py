"""Functions to extract model inference."""
import tensorflow_probability.python.edward2 as ed


def get_variable_dict(model_fn, X):
    """Return an OrderedDict of model variables.

    Args:
        model_fn: (function) A model function.
        X: (Tensor or ndarray) A Tensor of input variables

    Returns:
        (OrderedDict of tf.Variables) Return a ordered dictionary of variable names
            and corresponding tf.Variables
    """
    with ed.tape() as rv_dict:
        _ = model_fn(X)

    return rv_dict


def make_log_prob_fn(model_fn, X, y):
    """Makes a log likelihood function for MCMC training.

    Args:
        model_fn: (function) A model function.
        X: (Tensor or ndarray) A Tensor of input variables
        y: (Tensor or ndarray) A Tensor of response variables

    Returns:
        (function): a log likelihood function for MCMC training
    """
    bnn_log_joint = ed.make_log_joint_fn(model_fn)
    rv_names = list(get_variable_dict(model_fn, X).keys())

    def bnn_log_prob_fn(*rv_positional_args):
        rv_kwargs = dict(zip(rv_names, rv_positional_args))
        rv_kwargs.pop('y', None)

        return bnn_log_joint(X, y=y, **rv_kwargs)

    return bnn_log_prob_fn
