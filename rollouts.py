import numpy as np

def collect_and_process_rollouts(
    env_fn, policy, reward_fn, global_session,
    n_timesteps=10000, max_ep_len=500,
    discount=0.99, gae_lambda=0.95
):
    # collect n_timesteps of data from n_envs rollouts in parallel
    obs, next_obs, actions, env_rewards = [], [], [], []
    ep_lens = []

    n_envs = int(np.ceil(n_timesteps / max_ep_len))
    env_vec = [env_fn() for n in range(n_envs)]
    env_timesteps = [0 for n in range(n_envs)]
    obs_vec = [[env_vec[n].reset()] for n in range(n_envs)]
    next_obs_vec = [[] for n in range(n_envs)]
    actions_vec = [[] for n in range(n_envs)]
    env_rewards_vec = [[] for n in range(n_envs)]

    while len(obs) < n_timesteps:
        cur_obs = np.array([obs[-1] for obs in obs_vec])
        action_vec = policy.act(cur_obs, global_session)
        for n in reversed(range(n_envs)):
            action = action_vec[n]
            # threshold actions
            threshholded_action = np.clip(action, env_vec[n].action_space.low, env_vec[n].action_space.high)
            ob, env_reward, done, info = env_vec[n].step(threshholded_action)
            obs_vec[n].append(ob)
            next_obs_vec[n].append(ob)
            actions_vec[n].append(action)
            env_rewards_vec[n].append(env_reward)
            env_timesteps[n] += 1
            if done or env_timesteps[n] >= max_ep_len:
                # record data
                env_vec.pop(n)
                ep_obs = obs_vec.pop(n)[:-1]
                ep_lens.append(len(ep_obs))
                obs.extend(ep_obs)
                next_obs.extend(next_obs_vec.pop(n))
                actions.extend(actions_vec.pop(n))
                env_rewards.extend(env_rewards_vec.pop(n))
                env_timesteps.pop(n)
                # add new env
                env_vec.append(env_fn())
                obs_vec.append([env_vec[-1].reset()])
                next_obs_vec.append([])
                actions_vec.append([])
                env_rewards_vec.append([])
                env_timesteps.append(0)

    obs, next_obs, actions, env_rewards = np.array(obs), np.array(next_obs), np.array(actions), np.expand_dims(env_rewards, axis=1)

    # get action_probs, values, entropies for all timesteps
    action_log_probs, values, entropies = policy.rollout_data(obs, actions, global_session)
    action_log_probs, values, entropies = np.expand_dims(action_log_probs, axis=1), np.array(values), np.expand_dims(entropies, axis=1)

    # apply reward function
    rewards = reward_fn(obs, next_obs, actions, action_log_probs, env_rewards, values, entropies)
    print('avg_ep_reward:', sum(rewards) / len(ep_lens))

    # get value_targets and advantages
    value_targets, advantages = [], []
    start_ind = 0
    for ep_len in ep_lens:
        ep_rewards = rewards[start_ind : start_ind + ep_len]
        ep_values = values[start_ind : start_ind + ep_len]
        if ep_len < max_ep_len: # early termination
            last_value = 0
        else: # could still collect more rewards
            last_value = global_session.run(policy.values, feed_dict={policy.obs: [next_obs[start_ind+ep_len-1]]})[0, 0]
        ep_value_targets, ep_advantages = get_value_targets_and_advantages(ep_rewards, ep_values, last_value, discount=discount, gae_lambda=gae_lambda)
        value_targets.extend(ep_value_targets)
        advantages.extend(ep_advantages)
        start_ind += ep_len
    value_targets, advantages = np.array(value_targets), np.array(advantages)

    # can also apply advantage normalization per minibatch
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    return obs, next_obs, actions, action_log_probs, values, value_targets, advantages, rewards

def get_value_targets_and_advantages(
    rewards, values, last_value,
    discount=0.99, gae_lambda=0.95
):
    rollout_len = rewards.shape[0]
    value_targets = [[0] for _ in range(rollout_len)]
    advantages = [[0] for _ in range(rollout_len)]

    for t in reversed(range(rollout_len)):
        if t == rollout_len-1:
            value_targets[t][0] = rewards[t][0] + discount*last_value
            advantages[t][0] = rewards[t][0] + discount*last_value - values[t][0]
            # openai baselines version:
            # value_targets[t][0] = advantages[t][0] + values[t][0]
        else:
            value_targets[t][0] = rewards[t][0] + discount*values[t+1][0]
            advantages[t][0] = rewards[t][0] + discount*values[t+1][0] - values[t][0] \
                + discount*gae_lambda*advantages[t+1][0]
            # openai baselines version:
            # value_targets[t][0] = advantages[t][0] + values[t][0]

    # can try applying advantage normalization here
    # advantages = np.array(advantages)
    # advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    return value_targets, advantages
