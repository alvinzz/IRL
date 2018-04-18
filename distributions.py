import tensorflow as tf
import numpy as np

class DiagGaussian:
    def __init__(
        self,
        means,
        log_vars
    ):
        # means, stds are TF tensors of size (N, dim)
        self.means = means
        self.log_vars = log_vars
        self.log_stds = log_vars / 2.
        self.stds = tf.exp(self.log_stds)
        self.dim = tf.to_float(tf.shape(self.means)[-1])

    def log_prob(self, samples):
        zs = (samples - self.means) / self.stds
        return -tf.reduce_sum(self.log_stds, axis=1) \
            - 0.5 * tf.reduce_sum(tf.square(zs), axis=1) \
            - 0.5 * self.dim * np.log(2*np.pi)

    def sample(self):
        actions = self.means + self.stds * tf.random_normal(tf.shape(self.means))
        return actions

    def kl(self, other):
        assert isinstance(other, DiagGaussian)
        delta_means = self.means - other.means
        return tf.reduce_sum(other.log_stds - self.log_stds - 0.5 \
            + (tf.square(self.stds) + tf.square(delta_means)) \
                    / (2.0 * tf.square(other.stds)), \
            axis=1)

    def entropy(self):
        return tf.reduce_sum(self.log_stds \
            + 0.5 * np.log(2.0 * np.pi * np.e), \
            axis=1)
