from policies import GaussianMLPPolicy
from discriminators import AIRLDiscriminator
import tensorflow as tf
from rollouts import collect_and_process_rollouts

class AIRL:
    def __init__(self,
        name,
        env_fn, ob_dim, action_dim,
        reward_fn,
        expert_obs, expert_next_obs, expert_actions
    ):
        with tf.variable_scope(name):
            self.env_fn = env_fn
            self.ob_dim = ob_dim
            self.action_dim = action_dim

            self.expert_obs = expert_obs
            self.expert_next_obs = expert_next_obs
            self.expert_actions = expert_actions

            self.policy = GaussianMLPPolicy('policy', self.ob_dim, self.action_dim)
            if self.expert_obs is not None: # train expert only
                self.discriminator = AIRLDiscriminator('discriminator', self.ob_dim)
            self.reward_fn = reward_fn

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

    def train(self, n_iters, max_timesteps, max_ep_len, n_policy_gd_steps=4):
        for iter_ in range(n_iters):
            obs, next_obs, actions, action_probs, value_targets, advantages = collect_and_process_rollouts(self.env_fn(), self.policy, self.reward_fn, self.sess, max_timesteps, max_ep_len)
            for _  in range(n_policy_gd_steps):
                self.sess.run(self.policy.train_op, feed_dict={self.policy.obs: obs, self.policy.actions: actions, self.policy.old_action_log_probs: action_probs, self.policy.value_targets: value_targets, self.policy.advantages: advantages})
            if self.expert_obs is not None:
                self.discriminator.train(
                    self.expert_obs, self.expert_next_obs, self.expert_actions,
                    obs, next_obs, action_probs,
                    self.policy, self.sess
                )