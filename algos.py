from policies import GaussianMLPPolicy
from discriminators import AIRLDiscriminator
import tensorflow as tf
import numpy as np
from rollouts import collect_and_process_rollouts
from rewards import make_ent_env_reward_fn

class RL:
    def __init__(self,
        name,
        env_fn,
        checkpoint=None
    ):
        with tf.variable_scope(name):
            self.env_fn = env_fn
            self.ob_dim = env_fn().observation_space.shape[0]
            self.action_dim = env_fn().action_space.shape[0]

            self.policy = GaussianMLPPolicy('policy', self.ob_dim, self.action_dim)

            self.saver = tf.train.Saver()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

            if checkpoint:
                self.saver.restore(self.sess, checkpoint)

    def train(self, n_iters, max_timesteps=2000, max_ep_len=1000, reward_fn=make_ent_env_reward_fn(None)):
        for iter_ in range(n_iters):
            print('______________')
            print('Iteration', iter_)
            obs, next_obs, actions, action_log_probs, values, value_targets, advantages = collect_and_process_rollouts(self.env_fn, self.policy, reward_fn, self.sess, max_timesteps, max_ep_len)
            self.policy.optimizer.train(obs, next_obs, actions, action_log_probs, values, value_targets, advantages, self.sess)

class AIRL:
    def __init__(self,
        name,
        env_fn,
        expert_obs, expert_next_obs, expert_actions,
        checkpoint=None
    ):
        with tf.variable_scope(name):
            self.env_fn = env_fn
            self.ob_dim = env_fn().observation_space.shape[0]
            self.action_dim = env_fn().action_space.shape[0]

            self.expert_obs = expert_obs
            self.expert_next_obs = expert_next_obs
            self.expert_actions = expert_actions

            self.policy = GaussianMLPPolicy('policy', self.ob_dim, self.action_dim)
            self.discriminator = AIRLDiscriminator('discriminator', self.ob_dim)

            self.saver = tf.train.Saver()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

            if checkpoint:
                self.saver.restore(self.sess, checkpoint)

    def train(self, n_iters, max_timesteps=2000, max_ep_len=1000, reward_fn=make_ent_env_reward_fn(None)):
        # AIRL: keep replay buffer of past 20 iterations of policies
        obs_buffer, next_obs_buffer, action_log_probs_buffer = None, None, None
        for iter_ in range(n_iters):
            print('______________')
            print('Iteration', iter_)
            obs, next_obs, actions, action_log_probs, values, value_targets, advantages = collect_and_process_rollouts(self.env_fn, self.policy, reward_fn, self.sess, max_timesteps, max_ep_len)

            if obs_buffer is None:
                obs_buffer, next_obs_buffer, action_log_probs_buffer = obs, next_obs, action_log_probs
            else:
                obs_buffer, next_obs_buffer, action_log_probs_buffer = np.concatenate((obs_buffer, obs)), np.concatenate((next_obs_buffer, next_obs)), np.concatenate((action_log_probs_buffer, action_log_probs))
                obs_buffer, next_obs_buffer, action_log_probs_buffer = obs_buffer[-20*max_timesteps:], next_obs_buffer[-20*max_timesteps:], action_log_probs_buffer[-20*max_timesteps:]

            self.policy.optimizer.train(obs, next_obs, actions, action_log_probs, values, value_targets, advantages, self.sess)
            self.discriminator.train(
                self.expert_obs, self.expert_next_obs, self.expert_actions,
                obs_buffer, next_obs_buffer, action_log_probs_buffer,
                self.policy, self.sess
            )
