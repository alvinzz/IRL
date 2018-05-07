import numpy as np
from utils import create_intention_obs

def collect_and_process_rollouts(
    env_fn, policy, reward_fn, global_session,
    n_timesteps=10000, max_ep_len=500,
    discount=0.99, gae_lambda=0.95,
    shairl_timestep_normalization=False
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
    if not shairl_timestep_normalization:
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    else:
        ob_dim = obs.shape[1] - 1
        timesteps = obs[:, ob_dim]
        for timestep in np.arange(max_ep_len):
            timestep_inds = (timesteps == timestep)
            if np.any(timestep_inds):
                advantages[timestep_inds] = (advantages[timestep_inds] - np.mean(advantages[timestep_inds])) / (np.std(advantages[timestep_inds]) + 1e-8)

    return obs, next_obs, actions, action_log_probs, values, value_targets, advantages, rewards

def collect_and_process_intention_rollouts(
    env_fn, policy, reward_fn, n_intentions, global_session,
    n_timesteps=10000, max_ep_len=1000,
    discount=0.99, gae_lambda=0.95
):
    # collect n_timesteps of data from n_envs rollouts in parallel
    obs, intentions, actions, env_rewards = [], [], [], []
    ep_lens = []

    n_envs = int(np.ceil(n_timesteps / max_ep_len))
    env_vec = [env_fn() for n in range(n_envs)]
    env_timesteps = [0 for n in range(n_envs)]
    obs_vec = [[env_vec[n].reset()] for n in range(n_envs)]
    intentions_vec = [[] for n in range(n_envs)]
    actions_vec = [[] for n in range(n_envs)]
    env_rewards_vec = [[] for n in range(n_envs)]

    ob_dim = obs_vec[0][0].shape[0]

    while len(obs) < n_timesteps:
        cur_obs = np.array([np.concatenate((obs[-1], np.zeros(n_intentions))) for obs in obs_vec])
        intention_vec = [np.random.randint(n_intentions) for n in range(n_envs)]
        for n in range(n_envs):
            cur_obs[n][ob_dim + intention_vec[n]] = 1
        action_vec = policy.act(cur_obs, global_session)
        for n in reversed(range(n_envs)):
            action = action_vec[n]
            # threshold actions
            threshholded_action = np.clip(action, env_vec[n].action_space.low, env_vec[n].action_space.high)
            ob, env_reward, done, info = env_vec[n].step(threshholded_action)
            obs_vec[n].append(ob)
            intentions_vec[n].append(intention_vec[n])
            actions_vec[n].append(action)
            env_rewards_vec[n].append(env_reward)
            env_timesteps[n] += 1
            if done or env_timesteps[n] >= max_ep_len:
                # record data
                env_vec.pop(n)
                ep_obs = obs_vec.pop(n)[:-1]
                ep_lens.append(len(ep_obs))
                obs.extend(ep_obs)
                intentions.extend(intentions_vec.pop(n))
                actions.extend(actions_vec.pop(n))
                env_rewards.extend(env_rewards_vec.pop(n))
                env_timesteps.pop(n)
                # add new env
                env_vec.append(env_fn())
                obs_vec.append([env_vec[-1].reset()])
                actions_vec.append([])
                env_rewards_vec.append([])
                env_timesteps.append(0)

    obs, intentions, actions, env_rewards = np.array(obs), np.array(intentions), np.array(actions), np.expand_dims(env_rewards, axis=1)

    # get action_probs, values, entropies for all timesteps
    intention_obs = create_intention_obs(obs, intentions, n_intentions)
    action_log_probs, values, entropies = policy.rollout_data(intention_obs, actions, global_session)
    action_log_probs, values, entropies = np.expand_dims(action_log_probs, axis=1), np.array(values), np.expand_dims(entropies, axis=1)

    # apply reward function
    rewards = reward_fn(obs, intentions, actions, action_log_probs, env_rewards, values, entropies)
    print('avg_ep_reward:', sum(rewards) / len(ep_lens))

    # get value_targets and advantages
    value_targets, advantages = [], []
    start_ind = 0
    for ep_len in ep_lens:
        ep_rewards = rewards[start_ind : start_ind + ep_len]
        ep_values = values[start_ind : start_ind + ep_len]
        ep_value_targets, ep_advantages = get_value_targets_and_advantages(ep_rewards, ep_values, last_value=0, discount=discount, gae_lambda=gae_lambda)
        value_targets.extend(ep_value_targets)
        advantages.extend(ep_advantages)
        start_ind += ep_len
    value_targets, advantages = np.array(value_targets), np.array(advantages)

    # can also apply advantage normalization per minibatch
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    return obs, intentions, intention_obs, actions, action_log_probs, values, value_targets, advantages, rewards

def collect_and_process_intention_choice_rollouts(
    env_fn, n_intentions, intention_policy, policy, intention_reward_fn, reward_fn, global_session,
    n_timesteps=10000, max_ep_len=1000,
    discount=0.99, gae_lambda=0.95
):
    # collect n_timesteps of data from n_envs rollouts in parallel
    obs, next_obs, intentions, actions, env_rewards = [], [], [], [], []
    ep_lens = []

    n_envs = int(np.ceil(n_timesteps / max_ep_len))
    env_vec = [env_fn() for n in range(n_envs)]
    env_timesteps = [0 for n in range(n_envs)]
    obs_vec = [[env_vec[n].reset()] for n in range(n_envs)]
    intentions_vec = [[] for n in range(n_envs)]
    actions_vec = [[] for n in range(n_envs)]
    env_rewards_vec = [[] for n in range(n_envs)]

    ob_dim = obs_vec[0][0].shape[0]

    while len(obs) < n_timesteps:
        cur_obs = np.array([np.concatenate((obs[-1], np.zeros(n_intentions))) for obs in obs_vec])
        intention_vec = intention_policy.act([obs[-1] for obs in obs_vec], global_session)
        for n in range(n_envs):
            cur_obs[n][ob_dim + intention_vec[n]] = 1
        action_vec = policy.act(cur_obs, global_session)
        for n in reversed(range(n_envs)):
            action = action_vec[n]
            # threshold actions
            threshholded_action = np.clip(action, env_vec[n].action_space.low, env_vec[n].action_space.high)
            ob, env_reward, done, info = env_vec[n].step(threshholded_action)
            obs_vec[n].append(ob)
            intentions_vec[n].append(intention_vec[n])
            actions_vec[n].append(action)
            env_rewards_vec[n].append(env_reward)
            env_timesteps[n] += 1
            if done or env_timesteps[n] >= max_ep_len:
                # record data
                env_vec.pop(n)
                ep_obs = obs_vec.pop(n)
                ep_next_obs = ep_obs[1:]
                next_obs.extend(ep_next_obs)
                ep_obs = ep_obs[:-1]
                ep_lens.append(len(ep_obs))
                obs.extend(ep_obs)
                intentions.extend(intentions_vec.pop(n))
                actions.extend(actions_vec.pop(n))
                env_rewards.extend(env_rewards_vec.pop(n))
                env_timesteps.pop(n)
                # add new env
                env_vec.append(env_fn())
                obs_vec.append([env_vec[-1].reset()])
                actions_vec.append([])
                env_rewards_vec.append([])
                env_timesteps.append(0)

    obs, next_obs, intentions, actions, env_rewards = np.array(obs), np.array(next_obs), np.array(intentions), np.array(actions), np.expand_dims(env_rewards, axis=1)

    # get action_probs, values, entropies for all timesteps
    intention_obs = create_intention_obs(obs, intentions, n_intentions)
    intention_log_probs, intention_values, intention_entropies = intention_policy.rollout_data(obs, intentions, global_session)
    intention_log_probs, intention_values, intention_entropies = np.expand_dims(intention_log_probs, axis=1), np.array(intention_values), np.expand_dims(intention_entropies, axis=1)
    action_log_probs, values, entropies = policy.rollout_data(intention_obs, actions, global_session)
    action_log_probs, values, entropies = np.expand_dims(action_log_probs, axis=1), np.array(values), np.expand_dims(entropies, axis=1)

    # apply reward function
    intention_rewards = intention_reward_fn(obs, next_obs, actions, None, intentions, intention_values, intention_entropies)
    rewards = reward_fn(obs, next_obs, actions, None, intentions, values, entropies)
    print('avg intention policy reward:', sum(intention_rewards) / len(ep_lens))
    print('avg policy reward:', sum(rewards) / len(ep_lens))

    # get value_targets and advantages
    intention_value_targets, intention_advantages = [], []
    start_ind = 0
    for ep_len in ep_lens:
        ep_rewards = intention_rewards[start_ind : start_ind + ep_len]
        ep_values = intention_values[start_ind : start_ind + ep_len]
        ep_value_targets, ep_advantages = get_value_targets_and_advantages(ep_rewards, ep_values, last_value=0, discount=discount, gae_lambda=gae_lambda)
        intention_value_targets.extend(ep_value_targets)
        intention_advantages.extend(ep_advantages)
        start_ind += ep_len
    intention_value_targets, intention_advantages = np.array(intention_value_targets), np.array(intention_advantages)
    # can also apply advantage normalization per minibatch
    intention_advantages = (intention_advantages - np.mean(intention_advantages)) / (np.std(intention_advantages) + 1e-8)

    value_targets, advantages = [], []
    start_ind = 0
    for ep_len in ep_lens:
        ep_rewards = rewards[start_ind : start_ind + ep_len]
        ep_values = values[start_ind : start_ind + ep_len]
        ep_value_targets, ep_advantages = get_value_targets_and_advantages(ep_rewards, ep_values, last_value=0, discount=discount, gae_lambda=gae_lambda)
        value_targets.extend(ep_value_targets)
        advantages.extend(ep_advantages)
        start_ind += ep_len
    value_targets, advantages = np.array(value_targets), np.array(advantages)
    # can also apply advantage normalization per minibatch
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    intention_policy_data = [intentions, intention_log_probs, intention_values, intention_value_targets, intention_advantages, intention_rewards]
    policy_data = [actions, action_log_probs, values, value_targets, advantages, rewards]
    return obs, next_obs, intention_obs, intention_policy_data, policy_data

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
