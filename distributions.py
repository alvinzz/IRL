import tensorflow as tf
import numpy as np

class DiagGaussian:
    def __init__(
        self,
        means,
        log_vars
    ):
        # means, vars are TF tensors of size (N, dim)
        self.means = means
        self.log_vars = log_vars
        self.vars = tf.exp(self.log_vars)
        self.dim = tf.to_float(tf.shape(self.means)[-1])

    def log_prob(self, samples):
        zs = (samples - self.means) / self.vars
        return -tf.reduce_sum(self.log_vars, axis=1) \
            - 0.5 * tf.reduce_sum(tf.square(zs), axis=1) \
            - 0.5 * self.dim * np.log(2 * np.pi)

    def sample(self):
        actions = self.means + self.vars * tf.random_normal(tf.shape(self.means))
        return actions

    def kl(self, other):
        assert isinstance(other, DiagGaussian)
        delta_means = self.means - other.means
        return tf.reduce_sum(other.log_vars - self.log_vars - 0.5 \
            + (tf.square(self.vars) + tf.square(delta_means)) \
                    / (2.0 * tf.square(other.vars)), \
            axis=1)

    def entropy(self):
        return tf.reduce_sum(self.log_vars \
            + 0.5 * np.log(2.0 * np.pi * np.e), \
            axis=1)
