"""Utility TF functions for building a Feedforward Network"""
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def Dense(X, W, b, activation):
    if not activation:
        return tf.matmul(X, W) + b

    return activation(tf.matmul(X, W) + b)