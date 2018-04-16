import tensorflow as tf

class ClipPPO:
    def __init__(self,
        ob_dim, action_dim, policy,
        clip_param=0.2, max_grad_norm=0.5,
        optimizer=tf.train.AdamOptimizer, learning_rate=1e-3, optimizer_epsilon=1e-5
    ):
        self.optimizer = optimizer(learning_rate=learning_rate, epsilon=optimizer_epsilon)

        self.actions = tf.placeholder(tf.float32, shape=[None, action_dim], name='actions')
        self.old_action_probs = tf.placeholder(tf.float32, shape=[None, 1], name='old_action_probs')
        self.old_values = tf.placeholder(tf.float32, shape=[None, 1], name='old_values')
        self.value_targets = tf.placeholder(tf.float32, shape=[None, 1], name='value_targets')
        self.advantages = tf.placeholder(tf.float32, shape=[None, 1], name='advantages')

        self.policy = policy
        self.distribution = policy.distribution
        self.value = policy.value
        self.action_log_probs = self.distribution.log_prob(self.actions)

        # clipped policy loss
        self.action_prob_ratio = tf.exp(self.action_log_probs) / self.old_action_probs
        self.policy_loss = -self.action_prob_ratio * self.advantages
        self.clipped_policy_loss = -tf.clip_by_value(self.action_prob_ratio, 1-clip_param, 1+clip_param) * self.advantages
        self.surr_policy_loss = tf.reduce_mean(tf.maximum(self.policy_loss, self.clipped_policy_loss))

        # clipped value loss
        self.value_loss = tf.square(self.value_targets - self.value)
        self.clipped_value_loss = tf.square(self.value_targets \
            - (self.old_values + tf.clip_by_value(self.value - self.old_values, -clip_param, clip_param))
        )
        # min or max here?
        self.surr_value_loss = 0.5 * tf.reduce_mean(tf.minimum(self.value_loss, self.clipped_value_loss))

        # total loss
        self.loss = self.surr_policy_loss + self.surr_value_loss

        # gradients
        self.params = tf.trainable_variables()
        self.grads = tf.gradients(self.loss, self.params)
        self.grads, _ = tf.clip_by_global_norm(self.grads, 0.5)
        self.grads = list(zip(self.grads, self.params))
        self.train_op = self.optimizer.apply_gradients(self.grads)

    def train(self,
        obs, actions, action_probs, values, value_targets, advantages,
        global_session,
        n_iters=4
    ):
        for iter_ in range(n_iters):
            global_session.run(
                self.train_op,
                feed_dict={self.policy.obs: obs, self.actions: actions, self.old_action_probs: action_probs, self.old_values: values, self.value_targets: value_targets, self.advantages: advantages}
            )
