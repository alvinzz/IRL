import tensorflow as tf
from utils import batchify

class ClipPPO:
    def __init__(self,
        ob_dim, action_dim, policy,
        clip_param=0.2, max_grad_norm=0.5,
        optimizer=tf.train.AdamOptimizer, learning_rate=1e-3, optimizer_epsilon=1e-5
    ):
        self.optimizer = optimizer(learning_rate=learning_rate, epsilon=optimizer_epsilon)

        self.actions = tf.placeholder(tf.float32, shape=[None, action_dim], name='actions')
        self.old_action_probs = tf.placeholder(tf.float32, shape=[None, 1], name='old_action_probs')
        self.value_targets = tf.placeholder(tf.float32, shape=[None, 1], name='value_targets')
        self.advantages = tf.placeholder(tf.float32, shape=[None, 1], name='advantages')

        self.policy = policy
        self.distribution = self.policy.distribution
        self.value = self.policy.value
        self.action_log_probs = tf.expand_dims(self.distribution.log_prob(self.actions), axis=1)

        # clipped policy loss
        self.action_prob_ratio = tf.exp(self.action_log_probs) / self.old_action_probs
        self.policy_loss = -self.action_prob_ratio * self.advantages
        self.clipped_policy_loss = -tf.clip_by_value(self.action_prob_ratio, 1-clip_param, 1+clip_param) * self.advantages
        self.surr_policy_loss = tf.reduce_mean(tf.maximum(self.policy_loss, self.clipped_policy_loss))

        # value loss
        self.value_loss = tf.reduce_mean(tf.square(self.value_targets - self.value))

        # total loss
        self.loss = self.surr_policy_loss + self.value_loss

        # gradients
        self.params = tf.trainable_variables()
        self.grads = tf.gradients(self.loss, self.params)
        self.grads, _ = tf.clip_by_global_norm(self.grads, max_grad_norm)
        self.grads = list(zip(self.grads, self.params))
        self.train_op = self.optimizer.apply_gradients(self.grads)

    def train(self,
        obs, next_obs, actions, action_probs, values, value_targets, advantages,
        global_session,
        n_iters=4, n_batches=4
    ):
        pol_loss, val_loss, action_prob_ratio = global_session.run(
            [self.surr_policy_loss, self.value_loss, self.action_prob_ratio],
            feed_dict={self.policy.obs: obs, self.actions: actions, self.old_action_probs: action_probs, self.value_targets: value_targets, self.advantages: advantages}
        )
        print('pol_loss:', pol_loss)
        print('val_loss:', val_loss)
        data = [obs, actions, action_probs, value_targets, advantages]
        for iter_ in range(n_iters):
            batched_data = batchify(data, n_batches)
            for minibatch in batched_data:
                mb_obs, mb_actions, mb_action_probs, mb_value_targets, mb_advantages = minibatch
                global_session.run(
                    self.train_op,
                    feed_dict={self.policy.obs: mb_obs, self.actions: mb_actions, self.old_action_probs: mb_action_probs, self.value_targets: mb_value_targets, self.advantages: mb_advantages}
                )
        pol_loss, val_loss, action_prob_ratio = global_session.run(
            [self.surr_policy_loss, self.value_loss, self.action_prob_ratio],
            feed_dict={self.policy.obs: obs, self.actions: actions, self.old_action_probs: action_probs, self.value_targets: value_targets, self.advantages: advantages}
        )
        print('pol_loss:', pol_loss)
        print('val_loss:', val_loss)
