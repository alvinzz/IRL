from policies import GaussianMLPPolicy
from discriminators import *
import tensorflow as tf
import numpy as np
from rollouts import *
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

            self.policy = GaussianMLPPolicy('policy', self.ob_dim, self.action_dim, hidden_dims=[64, 64, 64], learn_vars=True)

            self.saver = tf.train.Saver()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

            if checkpoint:
                self.saver.restore(self.sess, checkpoint)

    def train(self, n_iters, reward_fn=make_ent_env_reward_fn(None), batch_timesteps=10000, max_ep_len=500):
        for iter_ in range(n_iters):
            print('______________')
            print('Iteration', iter_)
            obs, next_obs, actions, action_log_probs, values, value_targets, advantages, rewards = collect_and_process_rollouts(self.env_fn, self.policy, reward_fn, self.sess, batch_timesteps, max_ep_len)
            # var = 0.1 / (iter_ + 1)
            # log_var = np.log(var)
            # assign_op = self.policy.log_vars.assign(np.tile(log_var, [1, self.action_dim]))
            # self.sess.run(assign_op)
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

    def train(self, n_iters, reward_fn, batch_timesteps=10000, max_ep_len=500):
        # AIRL: keep replay buffer of past 20 iterations of policies
        obs_buffer, next_obs_buffer, actions_buffer = None, None, None
        for iter_ in range(n_iters):
            print('______________')
            print('Iteration', iter_)
            obs, next_obs, actions, action_log_probs, values, value_targets, advantages, rewards = collect_and_process_rollouts(self.env_fn, self.policy, reward_fn, self.sess, batch_timesteps, max_ep_len)

            if obs_buffer is None:
                obs_buffer, next_obs_buffer, actions_buffer = obs, next_obs, actions
            else:
                obs_buffer, next_obs_buffer, actions_buffer = np.concatenate((obs_buffer, obs)), np.concatenate((next_obs_buffer, next_obs)), np.concatenate((actions_buffer, actions))
                obs_buffer, next_obs_buffer, actions_buffer = obs_buffer[-20*batch_timesteps:], next_obs_buffer[-20*batch_timesteps:], actions_buffer[-20*batch_timesteps:]

            self.policy.optimizer.train(obs, next_obs, actions, action_log_probs, values, value_targets, advantages, self.sess)
            self.discriminator.train(
                self.expert_obs, self.expert_next_obs, self.expert_actions,
                obs_buffer, next_obs_buffer, actions_buffer,
                self.policy, self.sess
            )

class SHAIRL:
    def __init__(self,
        name, basis_size,
        env_fns, n_timesteps,
        expert_obs, expert_next_obs, expert_actions,
        checkpoint=None
    ):
        with tf.variable_scope(name):
            self.n_tasks = len(env_fns)
            self.n_timesteps = n_timesteps
            self.env_fns = env_fns
            self.ob_dim = env_fns[0]().observation_space.shape[0]-1 # last dimension is time
            self.action_dim = env_fns[0]().action_space.shape[0]

            self.expert_obs = expert_obs
            self.expert_next_obs = expert_next_obs
            self.expert_actions = expert_actions

            self.policies = [GaussianMLPPolicy('policy{}'.format(task), self.ob_dim+1, self.action_dim, hidden_dims=[64, 64, 64], learn_vars=True) for task in range(self.n_tasks)]
            self.discriminator = SHAIRLDiscriminator('discriminator', self.ob_dim, self.action_dim, self.n_tasks, self.n_timesteps, basis_size, hidden_dims=[128, 64, 64])

            self.saver = tf.train.Saver()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

            if checkpoint:
                self.saver.restore(self.sess, checkpoint)

    def train(self, n_iters, reward_fns, batch_timesteps=1000, ep_len=100):
        # SHAIRL: keep replay buffer of ALL past policies
        obs_buffer, next_obs_buffer, actions_buffer = [None for _ in range(self.n_tasks)], [None for _ in range(self.n_tasks)], [None for _ in range(self.n_tasks)]
        for iter_ in range(n_iters):
            print('______________')
            print('Iteration', iter_)
            if obs_buffer[0] is not None:
                self.discriminator.train(
                    self.expert_obs, self.expert_next_obs, self.expert_actions,
                    obs_buffer, next_obs_buffer, actions_buffer,
                    self.policies, self.sess, n_iters=10, min_loss=0, #min_loss=0.1*(n_iters - iter_)/n_iters
                )

            for task in range(self.n_tasks):
                print('Task', task)
                obs, next_obs, actions, action_log_probs, values, value_targets, advantages, rewards = collect_and_process_rollouts(self.env_fns[task], self.policies[task], reward_fns[task], self.sess, batch_timesteps, ep_len, shairl_timestep_normalization=True)
                i = 0
                while np.sum(rewards)/batch_timesteps < np.log(0.4) and i < 50:
                    # # anneal variance over training
                    # var = 1 / (iter_ + 1)
                    # log_var = np.log(var)
                    # assign_op = self.policies[task].log_vars.assign(np.tile(log_var, [1, self.action_dim]))
                    # self.sess.run(assign_op)
                    self.policies[task].optimizer.train(obs, next_obs, actions, action_log_probs, values, value_targets, advantages, self.sess, n_iters=10)
                    obs, next_obs, actions, action_log_probs, values, value_targets, advantages, rewards = collect_and_process_rollouts(self.env_fns[task], self.policies[task], reward_fns[task], self.sess, batch_timesteps, ep_len, shairl_timestep_normalization=True)
                    i += 1

                if obs_buffer[task] is None:
                    obs_buffer[task], next_obs_buffer[task], actions_buffer[task] = obs, next_obs, actions
                else:
                    obs_buffer[task], next_obs_buffer[task], actions_buffer[task] = np.concatenate((obs_buffer[task], obs)), np.concatenate((next_obs_buffer[task], next_obs)), np.concatenate((actions_buffer[task], actions))
                    # obs_buffer[task], next_obs_buffer[task], actions_buffer[task] = obs_buffer[task][-20*batch_timesteps:], next_obs_buffer[task][-20*batch_timesteps:], actions_buffer[task][-20*batch_timesteps:]

