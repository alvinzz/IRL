from policies import *
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

            self.policy = GaussianMLPPolicy('policy', self.ob_dim, self.action_dim)

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
            self.policy.optimizer.train(obs, actions, action_log_probs, values, value_targets, advantages, self.sess)

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

            self.policy.optimizer.train(obs, actions, action_log_probs, values, value_targets, advantages, self.sess)
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

            self.policies = [GaussianMLPPolicy('policy{}'.format(task), self.ob_dim+1, self.action_dim, learn_vars=False) for task in range(self.n_tasks)]
            self.discriminator = SHAIRLDiscriminator('discriminator', self.ob_dim, self.action_dim, self.n_tasks, self.n_timesteps, basis_size)

            self.saver = tf.train.Saver()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

            if checkpoint:
                self.saver.restore(self.sess, checkpoint)

    def train(self, n_iters, reward_fns, batch_timesteps=1000, ep_len=100, max_policy_iters=100):
        # SHAIRL: keep replay buffer of past 20 iterations of policies
        obs_buffer, next_obs_buffer, actions_buffer = [None for _ in range(self.n_tasks)], [None for _ in range(self.n_tasks)], [None for _ in range(self.n_tasks)]
        for iter_ in range(n_iters):
            print('______________')
            print('Iteration', iter_)
            for task in range(self.n_tasks):
                print('Task', task)
                # evaluate policy
                obs, next_obs, actions, action_log_probs, values, value_targets, advantages, rewards = collect_and_process_rollouts(self.env_fns[task], self.policies[task], reward_fns[task], self.sess, batch_timesteps, ep_len, shairl_timestep_normalization=True)
                avg_reward = np.mean(rewards)
                log_var = np.log(np.clip(1*(0.5 - np.exp(avg_reward)), 0.1, 1))
                # train until we fool discriminator over 50% of the time
                # give up after max_policy_iters iterations
                train_iters = 0
                while avg_reward < np.log(0.5) and train_iters < max_policy_iters:
                    assign_op = self.policies[task].log_vars.assign(np.tile(log_var, [1, self.action_dim]))
                    self.sess.run(assign_op)
                    self.policies[task].optimizer.train(obs, actions, action_log_probs, values, value_targets, advantages, self.sess)
                    obs, next_obs, actions, action_log_probs, values, value_targets, advantages, rewards = collect_and_process_rollouts(self.env_fns[task], self.policies[task], reward_fns[task], self.sess, batch_timesteps, ep_len, shairl_timestep_normalization=True)
                    # avg_reward = 0.6*avg_reward + 0.4*np.mean(rewards)
                    avg_reward = np.mean(rewards)
                    log_var = np.log(np.clip(1*(0.5 - np.exp(avg_reward)), 0.1, 1))
                    train_iters += 1

                if obs_buffer[task] is None:
                    obs_buffer[task], next_obs_buffer[task], actions_buffer[task] = obs, next_obs, actions
                else:
                    obs_buffer[task], next_obs_buffer[task], actions_buffer[task] = np.concatenate((obs_buffer[task], obs)), np.concatenate((next_obs_buffer[task], next_obs)), np.concatenate((actions_buffer[task], actions))
                    obs_buffer[task], next_obs_buffer[task], actions_buffer[task] = obs_buffer[task][-20*batch_timesteps:], next_obs_buffer[task][-20*batch_timesteps:], actions_buffer[task][-20*batch_timesteps:]

            self.discriminator.train(
                self.expert_obs, self.expert_next_obs, self.expert_actions,
                obs_buffer, next_obs_buffer, actions_buffer,
                self.policies, self.sess,
            )

class IntentionGAN:
    def __init__(self,
        name,
        env_fn,
        n_intentions,
        expert_obs, expert_actions,
        checkpoint=None
    ):
        with tf.variable_scope(name):
            self.env_fn = env_fn
            self.ob_dim = env_fn().observation_space.shape[0]
            self.action_dim = env_fn().action_space.shape[0]
            self.n_intentions = n_intentions

            self.expert_obs = expert_obs
            self.expert_actions = expert_actions

            self.policy = GaussianMLPPolicy('policy', self.ob_dim+self.n_intentions, self.action_dim)
            self.discriminator = StandardDiscriminator('discriminator', self.ob_dim, self.action_dim)
            self.intention_inferer = IntentionDiscriminator('intention_inferer', self.ob_dim, self.action_dim, self.n_intentions)

            self.saver = tf.train.Saver()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

            if checkpoint:
                self.saver.restore(self.sess, checkpoint)

    def train(self, n_iters, reward_fn, batch_timesteps=10000, max_ep_len=1000):
        # keep replay buffer
        obs_buffer, actions_buffer = None, None
        for iter_ in range(n_iters):
            print('______________')
            print('Iteration', iter_)
            obs, intentions, intention_obs, actions, action_log_probs, values, value_targets, advantages, rewards = collect_and_process_intention_rollouts(self.env_fn, self.policy, reward_fn, self.n_intentions, self.sess, batch_timesteps, max_ep_len)

            if obs_buffer is None:
                obs_buffer, actions_buffer = obs, actions
            else:
                obs_buffer, actions_buffer = np.concatenate((obs_buffer, obs)), np.concatenate((actions_buffer, actions))

            self.policy.optimizer.train(intention_obs, actions, action_log_probs, values, value_targets, advantages, self.sess)
            self.discriminator.train(
                self.expert_obs, self.expert_actions,
                obs_buffer, actions_buffer,
                self.policy, self.sess
            )
            self.intention_inferer.train(
                obs, actions, intentions, self.sess
            )

class IntentionChoiceGAN:
    def __init__(self,
        name,
        env_fn,
        n_intentions,
        expert_obs, expert_next_obs, expert_actions,
        checkpoint=None
    ):
        with tf.variable_scope(name):
            self.env_fn = env_fn
            self.ob_dim = env_fn().observation_space.shape[0]
            self.action_dim = env_fn().action_space.shape[0]
            self.n_intentions = n_intentions

            self.expert_obs = expert_obs
            self.expert_next_obs = expert_next_obs
            self.expert_actions = expert_actions

            self.intention_policy = CategoricalMLPPolicy('intention_policy', self.ob_dim, self.n_intentions, hidden_dims=[64, 64, 64])
            self.policy = GaussianMLPPolicy('policy', self.ob_dim+self.n_intentions, self.action_dim, hidden_dims=[])
            self.discriminator = IntentionAIRLDiscriminator('discriminator', self.ob_dim, self.n_intentions, hidden_dims=[64, 64, 64])
            self.intention_inferer = IntentionDiscriminator('intention_inferer', self.ob_dim, self.action_dim, self.n_intentions, hidden_dims=[64, 64])

            self.saver = tf.train.Saver()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

            if checkpoint:
                self.saver.restore(self.sess, checkpoint)

    def train(self, n_iters, intention_reward_fn, reward_fn, batch_timesteps=10000, max_ep_len=1000):
        # keep replay buffer
        obs_buffer, next_obs_buffer, actions_buffer = None, None, None
        for iter_ in range(n_iters):
            print('______________')
            print('Iteration', iter_)
            obs, next_obs, intention_obs, intention_policy_data, policy_data = collect_and_process_intention_choice_rollouts(self.env_fn, self.n_intentions, self.intention_policy, self.policy, intention_reward_fn, reward_fn, self.sess, batch_timesteps, max_ep_len)
            intentions, intention_log_probs, intention_values, intention_value_targets, intention_advantages, intention_rewards = intention_policy_data
            actions, action_log_probs, values, value_targets, advantages, rewards = policy_data

            if obs_buffer is None:
                obs_buffer, next_obs_buffer, actions_buffer = obs, next_obs, actions
            else:
                obs_buffer, next_obs_buffer, actions_buffer = np.concatenate((obs_buffer, obs)), np.concatenate((next_obs_buffer, next_obs)), np.concatenate((actions_buffer, actions))

            self.discriminator.train(
                self.expert_obs, self.expert_next_obs, self.expert_actions,
                obs_buffer, next_obs_buffer, actions_buffer,
                self.intention_policy, self.policy, self.sess, n_iters=1000
            )
            self.intention_inferer.train(
                obs, actions, intentions, self.sess, n_iters=10
            )
            # train for equal amount of time on each intention
            intention_list = np.arange(self.n_intentions)
            np.random.shuffle(intention_list)
            avg_rewards = np.zeros(4)
            for intention in intention_list:
                one_hot_intention = np.zeros(self.n_intentions)
                one_hot_intention[intention] = 1
                intention_inds = np.arange(intentions.shape[0])[intentions==intention]
                avg_rewards[intention] = np.mean(rewards[intention_inds]) if intention_inds.size else -20
                if intention_inds.size:
                    print('intention', intention, 'avg reward', np.mean(rewards[intention_inds]), '\n\tavg intention reward', np.mean(intention_rewards[intention_inds]), 'advantage', np.mean(intention_advantages[intention_inds]))

            train_iters = (np.exp(-avg_rewards) / np.sum(np.exp(-avg_rewards))).tolist()
            train_iters = [int(100*i) for i in train_iters]
            for intention in intention_list:
                self.policy.optimizer.train(intention_obs[intention_inds], actions[intention_inds], action_log_probs[intention_inds], values[intention_inds], value_targets[intention_inds], advantages[intention_inds], self.sess, n_iters=train_iters[intention])

            self.intention_policy.optimizer.train(obs, intentions, intention_log_probs, intention_values, intention_value_targets, intention_advantages, self.sess, n_iters=1)
            # counts = np.array([np.sum(intentions == intention) for intention in range(self.n_intentions)])
            # frequencies = counts / np.sum(counts)
            # train_iters = (np.exp(-frequencies) / np.sum(np.exp(-frequencies))).tolist()
            # train_iters = [int(10*i) for i in train_iters]
            # for intention in sorted(range(self.n_intentions), key=lambda i: frequencies[i], reverse=True):
            #     intention_inds = np.arange(intentions.shape[0])[intentions==intention]
            #     self.intention_policy.optimizer.train(obs[intention_inds], intentions[intention_inds], intention_log_probs[intention_inds], intention_values[intention_inds], intention_value_targets[intention_inds], intention_advantages[intention_inds], self.sess, n_iters=10)
