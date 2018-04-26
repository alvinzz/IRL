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
        hidden_dims=[64, 64, 64],
        hidden_activation=tf.nn.relu,
        weight_init=tf.contrib.layers.xavier_initializer,
        bias_init=tf.zeros_initializer,
        discount=0.99,
        learning_rate=1e-3
    ):
        self.last_loss = -np.log(0.5)
        self.noise_param = 0
        with tf.variable_scope(name):
            self.obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='obs')
            self.next_obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='next_obs')
            # reward network. assumes rewards are functions of state only
            self.reward_network = MLP('reward', ob_dim, 1, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
            self.rewards = self.reward_network.layers['out']
            # value network
            self.value_network = MLP('value', ob_dim, 1, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
            self.next_value_network = MLP('value', ob_dim, 1, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.next_obs, reuse=True)
            self.values = self.value_network.layers['out']
            self.next_values = self.next_value_network.layers['out']

            # estimate p(a | s, expert) as follows:
            self.expert_action_log_probs = self.rewards + discount*self.next_values - self.values

            self.policy_action_log_probs = tf.placeholder(tf.float32, shape=[None, 1], name='policy_action_log_probs')
            self.expert_log_probs = self.expert_action_log_probs - tf.reduce_logsumexp(tf.stack((self.expert_action_log_probs, self.policy_action_log_probs)), axis=0)

            # training
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels') # 1 if from expert else 0
            # self.expert_loss = -tf.log(self.expert_probs)
            # self.policy_loss = -tf.log(1-self.expert_probs)
            self.loss = -tf.reduce_mean(self.labels*self.expert_log_probs + (1-self.labels)*tf.log(1-tf.exp(self.expert_log_probs)+1e-8))
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def expert_log_prob(self, obs, next_obs, policy_action_log_probs, global_session):
        expert_log_probs = global_session.run(
            self.expert_log_probs,
            feed_dict={self.obs: obs, self.next_obs: next_obs, self.policy_action_log_probs: policy_action_log_probs}
        )
        return expert_log_probs

    def reward(self, obs, global_session):
        rewards = global_session.run(
            self.rewards,
            feed_dict={self.obs: obs}
        )
        return rewards

    def train(self,
            expert_obs, expert_next_obs, expert_actions,
            policy_obs, policy_next_obs, policy_actions,
            policy, global_session,
            n_iters=100, batch_size=32
        ):
            # add instance noise to the inputs if the signal for the generator is too weak
            if self.last_loss < -np.log(0.9):
                print('increasing noise param to weaken discriminator')
                self.noise_param += 0.01
            else:
                self.noise_param *= 0.9
            expert_obs += np.random.normal(loc=0, scale=self.noise_param*np.std(expert_obs, axis=0), size=expert_obs.shape)
            expert_next_obs += np.random.normal(loc=0, scale=self.noise_param*np.std(expert_next_obs, axis=0), size=expert_next_obs.shape)
            expert_actions += np.random.normal(loc=0, scale=self.noise_param*np.std(expert_actions, axis=0), size=expert_actions.shape)
            policy_obs += np.random.normal(loc=0, scale=self.noise_param*np.std(policy_obs, axis=0), size=policy_obs.shape)
            policy_next_obs += np.random.normal(loc=0, scale=self.noise_param*np.std(policy_next_obs, axis=0), size=policy_next_obs.shape)
            policy_actions += np.random.normal(loc=0, scale=self.noise_param*np.std(policy_actions, axis=0), size=policy_actions.shape)

            policy_action_log_probs = global_session.run(
                policy.action_log_probs,
                feed_dict={policy.obs: policy_obs, policy.actions: policy_actions}
            )
            policy_action_log_probs = np.expand_dims(policy_action_log_probs, axis=1)

            expert_action_log_probs_under_policy = global_session.run(
                policy.action_log_probs,
                feed_dict={policy.obs: expert_obs, policy.actions: expert_actions}
            )
            expert_action_log_probs_under_policy = np.expand_dims(expert_action_log_probs_under_policy, axis=1)

            labels = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))))

            for iter_ in range(n_iters):
                mb_expert_obs, mb_expert_next_obs, mb_expert_action_log_probs_under_policy = sample_minibatch(expert_obs, expert_next_obs, expert_action_log_probs_under_policy, batch_size)
                mb_policy_obs, mb_policy_next_obs, mb_policy_action_log_probs = sample_minibatch(policy_obs, policy_next_obs, policy_action_log_probs, batch_size)
                mb_obs = np.concatenate((mb_expert_obs, mb_policy_obs))
                mb_next_obs = np.concatenate((mb_expert_next_obs, mb_policy_next_obs))
                mb_policy_action_log_probs = np.concatenate((mb_expert_action_log_probs_under_policy, mb_policy_action_log_probs))
                global_session.run(
                    self.train_op,
                    feed_dict={self.obs: mb_obs, self.next_obs: mb_next_obs, self.policy_action_log_probs: mb_policy_action_log_probs, self.labels: labels}
                )

            self.last_loss = global_session.run(
                self.loss,
                feed_dict={self.obs: mb_obs, self.next_obs: mb_next_obs, self.policy_action_log_probs: mb_policy_action_log_probs, self.labels: labels}
            )
            print('discrim_loss:', self.last_loss)
