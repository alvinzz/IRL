import tensorflow as tf
import numpy as np
from distributions import DiagGaussian
from networks import MLP
from optimizers import ClipPPO

class GaussianMLPPolicy:
    def __init__(
        self,
        name,
        ob_dim,
        action_dim,
        var_network=False, # NN if true, else trainable params indep of obs
        out_activation=None,
        hidden_dims=[32, 32],
        hidden_activation=tf.nn.relu,
        weight_init=tf.contrib.layers.xavier_initializer,
        bias_init=tf.zeros_initializer,
        optimizer=ClipPPO
    ):
        with tf.variable_scope(name):
            self.obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='obs')

            # policy net
            self.mean_network = MLP('means', ob_dim, action_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)

            if var_network:
                self.log_var_network = MLP('log_vars', ob_dim, action_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
                self.log_var = self.log_var_network.layers['out']
            else:
                self.log_var_network = MLP('log_vars', ob_dim, action_dim, out_activation=out_activation, hidden_dims=[], hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
                self.log_var = self.log_var_network.layers['out']

            self.distribution = DiagGaussian(self.mean_network.layers['out'], self.log_var)

            # value net
            self.value_network = MLP('values', ob_dim, action_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
            self.value = self.value_network.layers['out']

            # training, PPO for now
            self.optimizer = optimizer(ob_dim, action_dim, self)

    def act(self, obs, global_session):
        action = global_session.run(
            self.distribution.sample(),
            feed_dict={self.obs: obs}
        )
        return action

    def rollout_data(self, obs, actions, global_session):
        action_probs, values, entropies = global_session.run(
            [tf.exp(self.distribution.log_prob(actions)), self.value, self.distribution.entropy()],
            feed_dict={self.obs: obs}
        )
        return action_probs, values, entropies
