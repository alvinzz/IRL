from policies import GaussianMLPPolicy
from discriminators import AIRLDiscriminator
import tensorflow as tf
from rollouts import collect_and_process_rollouts
from rewards import env_reward_fn

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
            if self.expert_obs is not None: # train expert only
                self.discriminator = AIRLDiscriminator('discriminator', self.ob_dim)

            self.saver = tf.train.Saver()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

            if checkpoint:
                self.saver.restore(self.sess, checkpoint)

    def train(self, n_iters, max_timesteps, max_ep_len, reward_fn=env_reward_fn):
        for iter_ in range(n_iters):
            print('______________')
            print('Iteration', iter_)
            obs, next_obs, actions, action_log_probs, values, value_targets, advantages = collect_and_process_rollouts(self.env_fn, self.policy, reward_fn, self.sess, max_timesteps, max_ep_len)
            self.policy.optimizer.train(obs, next_obs, actions, action_log_probs, values, value_targets, advantages, self.sess)
            if self.expert_obs is not None:
                self.discriminator.train(
                    self.expert_obs, self.expert_next_obs, self.expert_actions,
                    obs, next_obs, action_log_probs,
                    self.policy, self.sess
                )
