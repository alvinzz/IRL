import tensorflow as tf
import numpy as np
from rollouts import collect_and_process_rollouts
from distributions import DiagGaussian
from networks import MLP
from rewards import make_AIRL_reward_fn, env_reward_fn

class GaussianMLPPolicy:
    def __init__(
        self,
        name,
        ob_dim,
        action_dim,
        learn_var=True, # else use a constant variance of 1
        var_network=False, # NN if true, else affine fn
        value_network=False, # NN if true, else affine fn
        out_activation=None,
        hidden_dims=[32, 32],
        hidden_activation=tf.nn.relu,
        weight_init=tf.contrib.layers.xavier_initializer,
        bias_init=tf.zeros_initializer,
        learning_rate=1e-3
    ):
        with tf.variable_scope(name):
            self.obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='obs')

            # policy net
            self.mean_network = MLP('means', ob_dim, action_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)

            if not learn_var:
                self.log_var = tf.get_variable('log_vars', shape=[1, action_dim], initializer=tf.zeros_initializer())
            else:
                if var_network:
                    self.log_var_network = MLP('log_vars', ob_dim, action_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
                    self.log_var = self.log_var_network.layers['out']
                else:
                    self.log_var_network = MLP('log_vars', ob_dim, action_dim, out_activation=out_activation, hidden_dims=[], hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
                    self.log_var = self.log_var_network.layers['out']

            self.distribution = DiagGaussian(self.mean_network.layers['out'], self.log_var)

            # value net
            if value_network:
                self.value_network = MLP('values', ob_dim, action_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
            else:
                self.value_network = MLP('values', ob_dim, action_dim, out_activation=out_activation, hidden_dims=[], hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
            self.value = self.value_network.layers['out']

            # training, PPO for now
            # policy gradient with clipping
            self.actions = tf.placeholder(tf.float32, shape=[None, action_dim], name='actions')
            self.advantages = tf.placeholder(tf.float32, shape=[None, 1], name='advantages')
            self.action_log_probs = self.distribution.log_prob(self.actions)
            self.old_action_log_probs = tf.placeholder(tf.float32, shape=[None, 1], name='old_action_log_probs')

            self.action_prob_ratio = tf.exp(self.action_log_probs - self.old_action_log_probs)
            self.policy_loss = -self.action_prob_ratio * self.advantages
            self.clipped_policy_loss = -tf.clip_by_value(self.action_prob_ratio, 0.8, 1.2) * self.advantages
            self.surr_loss = tf.reduce_mean(tf.maximum(self.policy_loss, self.clipped_policy_loss))

            self.params = tf.trainable_variables()
            self.policy_grads = tf.gradients(self.surr_loss, self.params)
            self.policy_grads, _ = tf.clip_by_global_norm(self.policy_grads, 0.5)
            self.policy_grads = list(zip(self.policy_grads, self.params))
            self.policy_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)
            self.policy_train_op = self.policy_optimizer.apply_gradients(self.policy_grads)

            # train value fn
            self.value_targets = tf.placeholder(tf.float32, shape=[None, 1], name='value_targets')
            self.value_loss = tf.reduce_mean(tf.square(self.value_targets - self.value))
            self.value_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            self.value_train_op = self.value_optimizer.minimize(self.value_loss)

            self.train_op = (self.policy_train_op, self.value_train_op)

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
