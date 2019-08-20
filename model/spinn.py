from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import numbers
import argparse
from tensorflow.python.platform import tf_logging as logging


def group_lasso_regularizer(scale, scope=None):
    if isinstance(scale, numbers.Integral):
        raise ValueError('scale cannot be an integer: %s' % scale)
    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                             scale)
        if scale == 0.:
            logging.info('Scale of 0 disables regularizer.')
            return lambda _: None

    def group_lasso(weights, name=None):
        """Applies group regularization to weights."""
        with tf.name_scope(scope, 'group2_regularizer', [weights]) as name:
            my_scale = tf.convert_to_tensor(scale,
                                            dtype=weights.dtype.base_dtype,
                                            name='scale')
            return tf.multiply(
                my_scale,
                tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(weights), 1))),
                name=name)

    return group_lasso


def spinn_model(features, labels, mode, params):
    net = tf.feature_column.input_layer(features,
                                        params['feature_columns'])
    if params['hidden_units'][0] == 0:
        regularizer = tf.contrib.layers.l1_regularizer(scale=params['reg'])
        response = tf.layers.dense(net, params['n_response'],
                                   activation=None,
                                   kernel_regularizer=regularizer)
    else:
        regularizer = group_lasso_regularizer(scale=params['reg'])
        net = tf.layers.dense(net,
                              units=params['hidden_units'][0],
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer)
        if len(params['hidden_units']) >= 2:
            for units in params['hidden_units'][1:]:
                net = tf.layers.dense(net,
                                      units=units, activation=tf.nn.relu)
        response = tf.layers.dense(net, params['n_response'],
                                   activation=None)

    response = tf.squeeze(response)

    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "response": response,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    mse_loss = tf.losses.mean_squared_error(labels=labels,
                                            predictions=response)
    loss = tf.losses.get_total_loss(add_regularization_losses=True)

    # Compute evaluation metrics.
    mse = tf.metrics.mean_squared_error(labels=labels,
                                        predictions=response)
    metrics = {'MSE': mse}
    tf.summary.scalar("MSE", mse[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss,
                                  global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss, train_op=train_op)
