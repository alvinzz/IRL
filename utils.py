import numpy as np
import pickle
import gym
from envs import *
register_custom_envs()

def sample_minibatch(obs, next_obs, action_log_probs, batch_size):
    random_indices = np.random.randint(0, obs.shape[0], size=batch_size)
    return obs[random_indices], next_obs[random_indices], action_log_probs[random_indices]

def sample_basis_minibatch(obs, next_obs, action_log_probs, batch_size):
    n_tasks = len(obs)
    ob_dim = obs[0].shape[1] - 1
    mb_obs, mb_next_obs, mb_action_log_probs, mb_tasks_timesteps = [], [], [], []
    # sample minibatch for each task. Also split off time from obs, create tasks_timesteps array.
    for task in range(n_tasks):
        mb_task_obs, mb_task_next_obs, mb_task_action_log_probs = sample_minibatch(obs[task], next_obs[task], action_log_probs[task], batch_size)
        mb_task_obs, mb_task_tasks_timesteps, mb_task_next_obs = mb_task_obs[:, :ob_dim], np.concatenate((np.tile(task, (batch_size, 1)), mb_task_obs[:, ob_dim:]), axis=1), mb_task_next_obs[:, :ob_dim]
        mb_obs.extend(mb_task_obs)
        mb_next_obs.extend(mb_task_next_obs)
        mb_action_log_probs.extend(mb_task_action_log_probs)
        mb_tasks_timesteps.extend(mb_task_tasks_timesteps)
    mb_obs, mb_next_obs, mb_action_log_probs, mb_tasks_timesteps = np.array(mb_obs), np.array(mb_next_obs), np.array(mb_action_log_probs), np.array(mb_tasks_timesteps)
    return mb_obs, mb_next_obs, mb_action_log_probs, mb_tasks_timesteps

def batchify(data, batch_size):
    N = data[0].shape[0]
    # batch_size = int(np.ceil(N / n_batches))
    res = []
    random_inds = np.arange(N)
    np.random.shuffle(random_inds)
    start_ind = 0
    while start_ind < N:
        batch_inds = random_inds[start_ind : min(start_ind + batch_size, N)]
        res.append([category[batch_inds] for category in data])
        start_ind += batch_size
    return res

def collect_pointmass_expert_data():
    target_dict = {
        0: np.array([1, 0]),
        1: np.array([0, 1]),
        2: np.array([-1, 0]),
        3: np.array([0, -1]),
    }
    for i in range(4):
        for j in range(4):
            print(i, j)
            env = gym.make('PointMass-v{}{}'.format(i, j))
            obs, actions, next_obs = [], [], []
            for _ in range(20):
                obs.append(env.reset())
                for _ in range(20):
                    # env.render()
                    action = 16*(target_dict[i]-obs[-1][:2]) + np.random.normal(size=2)
                    ep_obs, reward, done, info = env.step(action)
                    # print(reward)
                    obs.append(ep_obs)
                    next_obs.append(ep_obs)
                    actions.append(action)
                for _ in range(20):
                    # env.render()
                    action = 16*(target_dict[j]-obs[-1][:2]) + np.random.normal(size=2)
                    ep_obs, reward, done, info = env.step(action)
                    # print(reward)
                    obs.append(ep_obs)
                    next_obs.append(ep_obs)
                    actions.append(action)
                obs = obs[:-1]
                # print(obs)
                # time.sleep(10)
            obs, next_obs, actions = np.array(obs), np.array(next_obs), np.array(actions)
            pickle.dump({'expert_obs': obs, 'expert_next_obs': next_obs, 'expert_actions': actions}, open('data/pointmass/expert-{}{}.pkl'.format(i, j), 'wb'))

if __name__ == '__main__':
    collect_pointmass_expert_data()
