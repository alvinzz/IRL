import tensorflow as tf
import numpy as np
from distributions import DiagGaussian
from networks import MLP
from optimizers import ClipPPO

import time

class GaussianMLPPolicy:
    def __init__(
        self,
        name,
        ob_dim,
        action_dim,
        learn_vars=True,
        var_network=False, # NN if true, else trainable params indep of obs
        out_activation=None,
        hidden_dims=[64, 64],
        hidden_activation=tf.nn.elu,
        weight_init=tf.contrib.layers.xavier_initializer,
        bias_init=tf.zeros_initializer,
        optimizer=ClipPPO
    ):
        with tf.variable_scope(name):
            self.obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='obs')

            # policy net
            self.mean_network = MLP('means', ob_dim, action_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
            self.means = self.mean_network.layers['out']

            if learn_vars:
                if var_network:
                    self.log_var_network = MLP('log_vars', ob_dim, action_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
                    self.log_vars = self.log_var_network.layers['out']
                else:
                    self.log_vars = tf.get_variable('log_vars', [1, action_dim], initializer=tf.zeros_initializer())
            else:
                self.log_vars = tf.get_variable('log_vars', trainable=False, initializer=tf.zeros_initializer())

            self.distribution = DiagGaussian(self.means, self.log_vars)
            self.sampled_actions = self.distribution.sample()

            self.actions = tf.placeholder(tf.float32, shape=[None, action_dim], name='actions')
            self.action_log_probs = self.distribution.log_prob(self.actions)
            self.entropies = self.distribution.entropy()

            # value net
            self.value_network = MLP('values', ob_dim, 1, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
            self.values = self.value_network.layers['out']

            # training, PPO for now
            self.optimizer = optimizer(ob_dim, action_dim, self)

    def act(self, obs, global_session):
        actions = global_session.run(
            self.sampled_actions,
            feed_dict={self.obs: obs}
        )
        return actions

    def test_act(self, obs, global_session):
        actions = global_session.run(
            self.means,
            feed_dict={self.obs: obs}
        )
        return actions

    def rollout_data(self, obs, actions, global_session):
        action_log_probs, values, entropies = global_session.run(
            [self.action_log_probs, self.values, self.entropies],
            feed_dict={self.obs: obs, self.actions: actions}
        )
        return action_log_probs, values, entropies

class CategoricalMLPPolicy:
    def __init__(
        self,
        name,
        ob_dim,
        action_dim,
        out_activation=None,
        hidden_dims=[64, 64],
        hidden_activation=tf.nn.elu,
        weight_init=tf.contrib.layers.xavier_initializer,
        bias_init=tf.zeros_initializer,
        optimizer=ClipPPO
    ):
        with tf.variable_scope(name):
            self.obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='obs')

            # policy net
            self.prob_network = MLP('probs', ob_dim, action_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
            self.unscaled_probs = self.prob_network.layers['out']
            self.probs = tf.nn.softmax(self.unscaled_probs)

            self.distribution = tf.contrib.distributions.Categorical(probs=self.probs)
            self.sampled_actions = self.distribution.sample()

            self.actions = tf.placeholder(tf.float32, shape=[None], name='actions')
            self.action_log_probs = tf.log(self.distribution.prob(self.actions) + 1e-8)
            self.entropies = self.distribution.entropy()

            # value net
            self.value_network = MLP('values', ob_dim, 1, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
            self.values = self.value_network.layers['out']

            # training, PPO for now
            self.optimizer = optimizer(ob_dim, action_dim, self)

    def act(self, obs, global_session):
        actions = global_session.run(
            self.sampled_actions,
            feed_dict={self.obs: obs}
        )
        return actions

    def test_act(self, obs, global_session):
        probs = global_session.run(
            self.probs,
            feed_dict={self.obs: obs}
        )
        return np.expand_dims(np.argmax(probs), axis=1)

    def test_probs(self, obs, global_session):
        probs = global_session.run(
            self.probs,
            feed_dict={self.obs: obs}
        )
        return probs

    def rollout_data(self, obs, actions, global_session):
        action_log_probs, values, entropies = global_session.run(
            [self.action_log_probs, self.values, self.entropies],
            feed_dict={self.obs: obs, self.actions: actions}
        )
        return action_log_probs, values, entropies
