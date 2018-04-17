import tensorflow as tf
import numpy as np
from networks import MLP
from utils import sample_minibatch

class AIRLDiscriminator:
    def __init__(
        self,
        name,
        ob_dim,
        out_activation=None,
        hidden_dims=[64, 64],
        hidden_activation=tf.nn.tanh,
        weight_init=tf.contrib.layers.xavier_initializer,
        bias_init=tf.zeros_initializer,
        discount=0.995,
        learning_rate=1e-3
    ):
        with tf.variable_scope(name):
            self.obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='obs')
            self.next_obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='next_obs')
            # reward network. assumes rewards are functions of state only
            self.reward_network = MLP('reward', ob_dim, 1, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
            # value network
            self.value_network = MLP('value', ob_dim, 1, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
            self.next_value_network = MLP('value', ob_dim, 1, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.next_obs, reuse=True)

            # estimate p(a | s, expert) as follows:
            self.log_expert_action_probs = self.reward_network.layers['out'] \
                + discount*self.next_value_network.layers['out'] \
                - self.value_network.layers['out']
            self.expert_action_probs = tf.exp(self.log_expert_action_probs)

            self.policy_action_probs = tf.placeholder(tf.float32, shape=[None, 1], name='policy_action_probs')
            self.expert_probs = self.expert_action_probs / (self.expert_action_probs + self.policy_action_probs) # assuming 50% of actions are from expert, and 50% from policy

            # training
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels') # 1 if from expert else 0
            # self.expert_loss = -tf.log(self.expert_probs)
            # self.policy_loss = -tf.log(1-self.expert_probs)
            self.loss = -tf.reduce_mean(tf.log((1-self.labels) + 2*(self.labels-0.5)*self.expert_probs))
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def expert_prob(self, obs, next_obs, policy_action_probs, global_session):
        expert_probs = global_session.run(
            self.expert_probs,
            feed_dict={self.obs: obs, self.next_obs: next_obs, self.policy_action_probs: policy_action_probs}
        )
        return expert_probs

    def train(self,
            expert_obs, expert_next_obs, expert_actions,
            policy_obs, policy_next_obs, policy_action_probs,
            policy, global_session,
            n_iters=100, batch_size=32
        ):
            expert_action_probs_under_policy = global_session.run(
                policy.action_probs,
                feed_dict={policy.obs: expert_obs}
            )
            expert_action_probs_under_policy = np.expand_dims(expert_action_probs_under_policy, axis=1)

            labels = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))))

            for iter_ in range(n_iters):
                mb_expert_obs, mb_expert_next_obs, mb_expert_action_probs_under_policy = sample_minibatch(expert_obs, expert_next_obs, expert_action_probs_under_policy, batch_size)
                mb_policy_obs, mb_policy_next_obs, mb_policy_action_probs = sample_minibatch(policy_obs, policy_next_obs, policy_action_probs, batch_size)
                mb_obs = np.concatenate((mb_expert_obs, mb_policy_obs))
                mb_next_obs = np.concatenate((mb_expert_next_obs, mb_policy_next_obs))
                mb_policy_action_probs = np.concatenate((mb_expert_action_probs_under_policy, mb_policy_action_probs))
                global_session.run(
                    self.train_op,
                    feed_dict={self.obs: mb_obs, self.next_obs: mb_next_obs, self.policy_action_probs: mb_policy_action_probs, self.labels: labels}
                )
