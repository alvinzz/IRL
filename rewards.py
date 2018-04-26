import numpy as np

def make_irl_reward_fn(model, env_reward_weight=0, entropy_weight=0.1, discriminator_reward_weight=1):
    def irl_reward(obs, next_obs, actions, action_log_probs, env_rewards, values, entropies):
        expert_log_probs = model.discriminator.expert_log_prob(obs, next_obs, action_log_probs, model.sess)
        reward = env_reward_weight*env_rewards \
            + entropy_weight*entropies \
            + discriminator_reward_weight*(expert_log_probs - np.log(1-np.exp(expert_log_probs)+1e-8))
        return reward
    return irl_reward

def make_env_reward_fn(model):
    def env_reward_fn(obs, next_obs, actions, action_log_probs, env_rewards, values, entropies):
        return env_rewards
    return env_reward_fn

def make_ent_env_reward_fn(model, entropy_weight=0.1):
    def ent_env_reward_fn(obs, next_obs, actions, action_log_probs, env_rewards, values, entropies):
        return env_rewards + entropy_weight*entropies
    return ent_env_reward_fn

def make_learned_reward_fn(model, entropy_weight=0.1):
    def discriminator_reward(obs, next_obs, actions, action_log_probs, env_rewards, values, entropies):
        reward = model.discriminator.reward(obs, model.sess) + entropy_weight*entropies
        return reward
    return discriminator_reward
