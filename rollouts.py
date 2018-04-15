import numpy as np

def rollout(env, policy, reward_fn, global_session, max_len=500):
    obs = []
    next_obs = []
    actions = []
    action_probs = []
    env_rewards = []
    values = []
    entropies = []
    obs.append(env.reset())

    done = False
    while len(obs) <= max_len and not done:
        action, action_prob, value, entropy = policy.act([obs[-1]], global_session)
        ob, env_reward, done, info = env.step(action)
        obs.append(ob.flatten())
        next_obs.append(ob.flatten())
        actions.append(action[0])
        action_probs.append(action_prob)
        env_rewards.append(env_reward[0])
        values.append(value[0, 0])
        entropies.append(entropy[0])
    obs = obs[:-1]

    rewards = reward_fn(obs, next_obs, actions, action_probs, env_rewards, values, entropies)
    return obs, next_obs, actions, action_probs, rewards, values

def collect_and_process_rollouts(
        env, policy, reward_fn, global_session,
        n_timesteps=10000, max_ep_len=500,
        discount=0.99, gae_lambda=0.98):
    # collect data and perform single-episode preprocessing
    obs, next_obs, actions, action_probs, advantages, value_targets = [], [], [], [], [], []
    avg_reward, n_eps = 0, 0
    while len(obs) < n_timesteps:
        ep_obs, ep_next_obs, ep_actions, ep_action_probs, ep_rewards, ep_values = rollout(env, policy, reward_fn, global_session, max_len=max_ep_len)
        # get value targets and calculate advantages with GAE
        ep_value_targets, ep_advantages = get_value_targets_and_advantages(ep_rewards, ep_values, discount=discount, gae_lambda=gae_lambda)
        obs, next_obs, actions, action_probs, value_targets, advantages = obs+ep_obs, next_obs+ep_next_obs, actions+ep_actions, action_probs+ep_action_probs, value_targets+ep_value_targets, advantages+ep_advantages
        avg_reward += sum(ep_rewards)
        n_eps += 1
    print('avg_ep_reward:', avg_reward / n_eps)

    # convert to numpy
    obs, next_obs, actions, action_probs, value_targets, advantages = np.array(obs), np.array(next_obs), np.array(actions), np.array(action_probs), np.array(value_targets), np.array(advantages)

    # normalize advantages
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    return obs, next_obs, actions, action_probs, value_targets, advantages

def get_value_targets_and_advantages(rewards, values,
        discount=0.99, gae_lambda=0.98):
    rollout_len = rewards.shape[0]
    value_targets = [[0] for _ in range(rollout_len)]
    advantages = [[0] for _ in range(rollout_len)]

    for t in reversed(range(rollout_len)):
        if t == rollout_len-1:
            value_targets[t][0] = rewards[t]
            advantages[t][0] = rewards[t] - values[t]
        else:
            value_targets[t][0] = rewards[t] + discount*values[t+1]
            advantages[t][0] = rewards[t] + discount*values[t+1] - values[t] \
                + discount*gae_lambda*advantages[t+1][0]

    return value_targets, advantages
