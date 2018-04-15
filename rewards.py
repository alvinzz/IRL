import numpy as np

def make_AIRL_reward_fn(discriminator, global_session, env_reward_weight=0, entropy_weight=0.1, discriminator_reward_weight=1):
    def AIRL_reward(obs, next_obs, actions, action_probs, env_rewards, values, entropies):
        expert_probs = discriminator.expert_prob(obs, next_obs, action_probs, global_session)
        return env_reward_weight*np.array(env_rewards) \
            + entropy_weight*np.array(entropies) \
            + discriminator_reward_weight*(np.log(expert_probs+1e-8) - np.log(1-expert_probs+1e-8)).flatten()
    return AIRL_reward

def env_reward_fn(obs, next_obs, actions, action_probs, env_rewards, values, entropies):
    return np.array(env_rewards)
