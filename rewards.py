import numpy as np
from scipy.stats import entropy

def make_irl_reward_fn(model, env_reward_weight=0, entropy_weight=0.1, discriminator_reward_weight=1):
    def irl_reward(obs, next_obs, actions, action_log_probs, env_rewards, values, entropies):
        expert_log_probs = model.discriminator.expert_log_prob(obs, next_obs, action_log_probs, model.sess)
        reward = env_reward_weight*env_rewards \
            + entropy_weight*entropies \
            + discriminator_reward_weight*expert_log_probs
        return reward
    return irl_reward

def make_shairl_reward_fn(model, task, env_reward_weight=0, entropy_weight=0.1, discriminator_reward_weight=1):
    def shairl_reward(obs, next_obs, actions, action_log_probs, env_rewards, values, entropies):
        expert_log_probs = model.discriminator.expert_log_prob(obs, next_obs, action_log_probs, task, model.sess)
        reward = env_reward_weight*env_rewards \
            + entropy_weight*entropies \
            + discriminator_reward_weight*expert_log_probs
        return reward
    return shairl_reward

def make_shairl_reward_fns(model, env_reward_weight=0, entropy_weight=0, discriminator_reward_weight=1):
    reward_fns = [make_shairl_reward_fn(model, task, env_reward_weight, entropy_weight, discriminator_reward_weight) for task in range(model.n_tasks)]
    return reward_fns

def make_intention_reward_fn(model, discriminator_reward_weight=1, intention_weight=7):
    def intention_reward(obs, next_obs, actions, action_log_probs, intentions, values, entropies):
        expert_log_probs = model.discriminator.expert_log_prob(obs, actions, model.sess)
        intention_probs = model.intention_inferer.intention_prob(obs, actions, model.sess)
        reward = discriminator_reward_weight*(expert_log_probs-np.log(1-np.exp(expert_log_probs)+1e-8)) \
            + intention_weight*np.log(np.expand_dims(np.choose(intentions, intention_probs.T), axis=1)+1e-8)
        return reward
    return intention_reward

def make_intention_chooser_reward_fn(model, n_intentions=4, diversity_weight=1, discriminator_reward_weight=1, commit_weight=0):
    def intention_chooser_reward(obs, next_obs, actions, action_log_probs, intentions, intention_values, intention_entropies):
        expert_log_probs = model.discriminator.expert_log_prob(obs, actions, model.sess)
        print('discrim', np.mean(expert_log_probs-np.log(1-np.exp(expert_log_probs)+1e-8)))
        counts = np.array([np.sum(intentions == intention) for intention in range(n_intentions)])
        frequencies = counts / np.sum(counts)
        # diversity_bonus = (0.25*np.ones(n_intentions) - frequencies)**3
        diversity_bonus = 1/(frequencies + 1e-8)
        diversity_reward = np.zeros(intentions.shape[0])
        for intention in range(n_intentions):
            diversity_reward[intentions == intention] = diversity_bonus[intention]
        diversity_reward = np.expand_dims(diversity_reward, axis=1)
        print('intention frequencies', frequencies)
        print('diversity bonus', diversity_weight*diversity_bonus)
        change_intentions = np.concatenate(([0], intentions[1:] != intentions[:-1]))
        change_intentions = np.expand_dims(change_intentions, axis=1)
        print('commit', np.mean(commit_weight*change_intentions))
        reward = diversity_weight*diversity_reward \
            + discriminator_reward_weight*(expert_log_probs-np.log(1-np.exp(expert_log_probs)+1e-8)) \
            - commit_weight*change_intentions
        return reward
    return intention_chooser_reward

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

def make_shairl_learned_reward_fn(model, task, entropy_weight=0.1):
    def discriminator_reward(obs, next_obs, actions, action_log_probs, env_rewards, values, entropies):
        reward = model.discriminator.reward(obs, task, model.sess) + entropy_weight*entropies
        return reward
    return discriminator_reward
