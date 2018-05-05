import tensorflow as tf
import numpy as np
from utils import batchify

class ClipPPO:
    def __init__(self,
        ob_dim, action_dim, policy,
        clip_param=0.1, max_grad_norm=0.1,
        optimizer=tf.train.AdamOptimizer, learning_rate=1e-4, optimizer_epsilon=1e-5
    ):
        self.optimizer = optimizer(learning_rate=learning_rate, epsilon=optimizer_epsilon)

        self.old_action_log_probs = tf.placeholder(tf.float32, shape=[None, 1], name='old_action_log_probs')
        self.old_values = tf.placeholder(tf.float32, shape=[None, 1], name='old_values')
        self.value_targets = tf.placeholder(tf.float32, shape=[None, 1], name='value_targets')
        self.advantages = tf.placeholder(tf.float32, shape=[None, 1], name='advantages')

        self.policy = policy
        self.obs = self.policy.obs
        self.distribution = self.policy.distribution
        self.actions = self.policy.actions
        self.action_log_probs = self.policy.action_log_probs
        self.values = self.policy.values
        self.log_vars = self.policy.log_vars

        # clipped policy loss
        self.action_prob_ratio = tf.exp(tf.expand_dims(self.action_log_probs, axis=1) - self.old_action_log_probs)
        self.policy_loss = -self.action_prob_ratio * self.advantages
        self.clipped_policy_loss = -tf.clip_by_value(self.action_prob_ratio, 1-clip_param, 1+clip_param) * self.advantages
        self.surr_policy_loss = tf.reduce_mean(tf.maximum(self.policy_loss, self.clipped_policy_loss))

        # value loss
        self.value_loss = tf.square(self.value_targets - self.values)
        self.clipped_value_loss = tf.square(
            self.value_targets - (self.old_values + tf.clip_by_value(self.values - self.old_values, -clip_param, clip_param))
        )
        # self.surr_value_loss = 0.5 * tf.reduce_mean(tf.maximum(self.value_loss, self.clipped_value_loss))
        self.surr_value_loss = 0.5 * tf.reduce_mean(self.value_loss)

        # total loss
        self.loss = self.surr_policy_loss + self.surr_value_loss

        # gradients
        self.params = tf.trainable_variables()
        self.grads = tf.gradients(self.loss, self.params)
        self.grads, _ = tf.clip_by_global_norm(self.grads, max_grad_norm)
        self.grads = list(zip(self.grads, self.params))
        self.train_op = self.optimizer.apply_gradients(self.grads)

    def train(self,
        obs, next_obs, actions, action_log_probs, values, value_targets, advantages,
        global_session,
        n_iters=10, batch_size=64
    ):
        data = [obs, actions, action_log_probs, values, value_targets, advantages]
        for iter_ in range(n_iters):
            batched_data = batchify(data, batch_size)
            for minibatch in batched_data:
                mb_obs, mb_actions, mb_action_log_probs, mb_values, mb_value_targets, mb_advantages = minibatch
                # normalize advantages here?
                # mb_advantages = (mb_advantages - np.mean(mb_advantages)) / (np.std(mb_advantages) + 1e-8)
                global_session.run(
                    self.train_op,
                    feed_dict={self.obs: mb_obs, self.actions: mb_actions, self.old_action_log_probs: mb_action_log_probs, self.old_values: mb_values, self.value_targets: mb_value_targets, self.advantages: mb_advantages}
                )
