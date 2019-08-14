"""Functions for defining BNN Models."""
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.edward2 as ed

import util.nn as net_util
import util.dtype as dtype_util


def define_bnn(n_node=15, n_layer=1,
               activation=tf.nn.relu,
               hidden_weight_sd=0.1,
               output_weight_sd=None,
               **kwargs):
    """Generates a model function for BNN regression model.

    Adapted from below Colab:
        https://colab.research.google.com/drive/1bWQcuR5gaBPpow6ARKnPPL-dtf2EvTae

        Priors:
            hidden_biases ~ N(0, weight_prior_sd)
            output_biases ~ N(0, weight_prior_sd)

            hidden_weights ~ N(0, weight_prior_sd)
            hidden_weights ~ N(0, 1 / n_node)

    Args:
        n_node: (int) Number of hidden nodes per layer.
        n_layer: (int) Number of hidden layers.
        activation: (function) An activation function. Default to ReLU
        hidden_weight_sd: (float) Standard deviation for the hidden weight prior.
        output_weight_sd: (float) Standard deviation for the output weight prior.
        kwargs: additional keywork arguments for compatibility purpose.

    Returns:
        model (function): A function that returns a output node.

    Raises:
        (ValueError): If n_layer is not positive integer.
    """
    if not isinstance(n_layer, int) or n_layer < 0:
        raise ValueError("'n_layer' must be a positive integer.")

    if not output_weight_sd:
        output_weight_sd = tf.sqrt(1 / n_node)

    # define model
    def model(X, sample_bias=False, bias_sample_range=None):
        """Defines a Bayesian Neural Network for regression

        Note:
            Output weight prior variance is set to: 10/K

        Args:
            # X: (np.ndarray of NP_DTYPE)  A matrix of input features between (0, 1),
            #     shape (n, d).
            # weight_list: (list of tf.Tensor) A list of hidden weight Tensors,
            #     each element has dtype = TF_DTYPE, and shape is (n_feature, n_node)
            #     (if input weight), (n_node, n_node) if hidden weight,
            #     and (n_node, 1) if output weight
            # bias_list: (list of tf.Tensor) A list of bias term for each layer,
            #     each element has dtype = TF_DTYPE, shape = (,).
            bias_sample_range: (list of int or None) the index of covariate to sample bias for.
                If `None` then sample only the first.

        Returns:
            (tf.Tensor) The output distribution.
        """

        # define architecture
        X = tf.convert_to_tensor(X, dtype=dtype_util.TF_DTYPE)
        n_sample, n_feature = X.shape.as_list()
        layer_size = [n_feature] + [n_node] * n_layer + [1]

        # intialize model building
        weight_list = []
        bias_list = []

        # input layer
        # input = tf.get_variable(initializer=X,
        #                         trainable=False,
        #                         dtype=dtype_util.TF_DTYPE,
        #                         name="input")
        input = X
        net = input

        # for (layer_id, (weights, biases)) in enumerate(zip(weight_list[:-1], bias_list[:-1])):
        #     with tf.variable_scope(scope_list[layer_id].original_name_scope):
        #         net = net_util.Dense(net, weights, biases, activation=activation)

        # hidden layers
        for layer_id in range(len(layer_size) - 1):
            with tf.variable_scope("layer_{}".format(layer_id), reuse=True):
                # configure hidden weight
                weight_shape = (layer_size[layer_id], layer_size[layer_id + 1])
                weight_scale = hidden_weight_sd if layer_id < n_layer else output_weight_sd

                # define random variables
                bias_rv = ed.Normal(loc=0., scale=hidden_weight_sd,
                                    name="bias_{}".format(layer_id))
                weight_rv = ed.Normal(loc=0.,
                                      scale=tf.ones(shape=weight_shape) * weight_scale,
                                      name="weight_{}".format(layer_id))
                # add to list for easy access
                bias_list += [bias_rv]
                weight_list += [weight_rv]

                # produce output, optionally, store output-layer hidden nodes
                if sample_bias:
                    if layer_id == n_layer:
                        phi = net  # shape (n_sample, n_node)
                net = net_util.Dense(net, weight_rv, bias_rv,
                                     activation=None if layer_id == n_layer else activation)

        # final output layer
        with tf.variable_scope("output"):
            # produce output prediction
            y_mean = net[:, 0]  # shape (n, ) (i.e., the number of data samples)
            std_devs = 1.

            # estimate variable importance (i.e. squared gradient norm)
            y_mean_grad = tf.gradients(y_mean, X)[0]
            var_imp = tf.reduce_mean(y_mean_grad ** 2, axis=0)

            # estimate variable importance bias
            if sample_bias:
                phi_grad = tf.vectorized_map(  # shape (n_node, n_sample, n_feature)
                    lambda phi_k: tf.gradients(phi_k, X)[0], tf.transpose(phi))
                phi_grad2 = tf.vectorized_map(  # shape (n_feature, n_node, n_node)
                    lambda dphi: tf.matmul(dphi, dphi, transpose_b=True),
                    tf.transpose(phi_grad, [2, 0, 1]))

                phi2_inv = tfp.math.pinv(  # shape (n_node, n_node)
                    tf.matmul(phi, phi, transpose_a=True))

                var_imp_mat = tf.tensordot(  # shape (n_feature, n_node, n_node)
                    phi_grad2, phi2_inv, axes=[2, 0])
                var_imp_bias = tf.vectorized_map(
                    tf.linalg.trace, var_imp_mat)

                # phi_grad = tf.stack(  # shape (n_node, n_sample, n_feature)
                #     [tf.gradients(phi[:, k], X)[0] for k in range(n_node)])
                # phi_inv = tfp.math.pinv(phi)  # shape (n_node, n_sample)
                #
                # var_imp_bias = tf.stack([
                #     tf.reduce_sum(tf.matmul(
                #         phi_grad[:, :, p], phi_inv, transpose_a=True) ** 2)
                #     for p in range(n_feature)])
                #
                var_imp_bias = var_imp_bias / n_sample
            else:
                var_imp_bias = None

            y = ed.Normal(loc=y_mean, scale=std_devs, name="y")

        return y, y_mean, var_imp, var_imp_bias, weight_list, bias_list

    return model


if __name__ == "__main__":
    import time

    import numpy as np
    import tensorflow as tf

    n_node = 50
    n_layer = 1
    activation = tf.nn.relu
    hidden_weight_sd = 0.1
    output_weight_sd = 0.1

    # investigate eigen behavior
    phi = bias_sample[0]['phi']
    print(np.linalg.svd(phi)[1])

    bias_sample = bias_sample[0]
    n_node, n_sample, n_feature = bias_sample['phi_grad'].shape

    beta_hat = bias_sample['phi_inv'].dot(y_train)
    beta_eps = bias_sample['phi_inv'].dot(np.random.normal(size=n_sample)) * np.std(f_train)

    sns.distplot(beta_hat)
    sns.distplot(beta_eps)

    var_imp_est = np.array(
        [np.mean((bias_sample['phi_grad'][:, :, i].T).dot(
            beta_hat) ** 2) for i in range(n_feature)])
    var_imp_eps = np.array(
        [np.mean((bias_sample['phi_grad'][:, :, i].T).dot(
            beta_eps) ** 2) for i in range(n_feature)])
    var_imp_eps2 = np.array(
        [np.sum(((bias_sample['phi_grad'][:, :, i].T).dot(
            bias_sample['phi_inv'])) ** 2) * np.var(f_train) / n_sample
         for i in range(n_feature)])

    print(imp_sample)
    print(var_imp_est)
    print(var_imp_eps)
    print(var_imp_eps2)
    print(bias_sample['bias'])
    print(bias_sample['bias_adj'])

    dPhi = bias_sample['phi_grad'][:, :, 10].T  # (N, K)
    Phi_inv = bias_sample['phi_inv']  # (K, N)
    dPhi_Phi_inv = dPhi.dot(Phi_inv)
    n_node, n_sample,  = Phi_inv.shape

    np.sum(dPhi_Phi_inv ** 2)/n_sample
    np.mean(dPhi_Phi_inv.dot(np.random.normal(size=n_sample)) ** 2)
