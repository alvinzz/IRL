import gym
from algos import RL, AIRL
from rewards import make_env_reward_fn, make_ent_env_reward_fn, make_irl_reward_fn, make_learned_reward_fn
from rollouts import collect_and_process_rollouts
import tensorflow as tf
import numpy as np
import pickle
from envs import *
register_custom_envs()
import time
import matplotlib.pyplot as plt

def train_expert(
    n_iters, save_dir, name,
    env_name, make_reward_fn=make_ent_env_reward_fn, irl_model_algo=AIRL, irl_model_name=None,
    timesteps_per_rollout=10000, ep_max_len=500, demo_timesteps=1e5,
    rl_algo=RL, use_checkpoint=False,
):
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    if use_checkpoint:
        checkpoint = '{}/{}_model'.format(save_dir, name)
    else:
        checkpoint = None
    expert_model = rl_algo(name, env_fn, checkpoint=checkpoint)
    if irl_model_name:
        irl_graph = tf.Graph()
        with irl_graph.as_default():
            reward_fn_model = irl_model_algo(irl_model_name, env_fn, None, None, None, checkpoint='{}/{}_model'.format(save_dir, irl_model_name))
    else:
        reward_fn_model = None

    print('\nTraining expert...')
    expert_model.train(n_iters, timesteps_per_rollout, ep_max_len, reward_fn=make_reward_fn(reward_fn_model))

    print('\nCollecting expert trajectories, evaluating on original task...')
    expert_obs, expert_next_obs, expert_actions, _, _, _, _ = collect_and_process_rollouts(env_fn, expert_model.policy, make_env_reward_fn(None), expert_model.sess, demo_timesteps, ep_max_len)
    pickle.dump({'expert_obs': expert_obs, 'expert_next_obs': expert_next_obs, 'expert_actions': expert_actions}, open('{}/{}.pkl'.format(save_dir, name), 'wb'))

    expert_model.saver.save(expert_model.sess, '{}/{}_model'.format(save_dir, name))
    return expert_model

def train_irl(
    n_iters, save_dir, name, expert_name,
    env_name, make_reward_fn=make_irl_reward_fn,
    timesteps_per_rollout=10000, ep_max_len=500,
    irl_algo=AIRL, use_checkpoint=False,
):
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    data = pickle.load(open('{}/{}.pkl'.format(save_dir, expert_name), 'rb'))
    expert_obs, expert_next_obs, expert_actions = data['expert_obs'], data['expert_next_obs'], data['expert_actions']

    print('\nTraining IRL...')
    if use_checkpoint:
        checkpoint = '{}/{}_model'.format(save_dir, name)
    else:
        checkpoint = None
    irl_model = irl_algo(name, env_fn, expert_obs, expert_next_obs, expert_actions, checkpoint=checkpoint)
    irl_model.train(n_iters, timesteps_per_rollout, ep_max_len, reward_fn=make_irl_reward_fn(irl_model))

    # evaluate and save model
    print('\nEvaluating policy on original task...')
    collect_and_process_rollouts(env_fn, irl_model.policy, make_env_reward_fn(None), irl_model.sess, 20*ep_max_len, ep_max_len)

    irl_model.saver.save(irl_model.sess, '{}/{}_model'.format(save_dir, name))
    return irl_model

def visualize_expert(env_name, expert_dir, expert_name, rl_algo=RL, ep_max_len=100, n_runs=1):
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    expert_model = rl_algo(expert_name, env_fn, checkpoint='{}/{}_model'.format(expert_dir, expert_name))
    env = gym.make(env_name)
    for n in range(n_runs):
        obs = env.reset()
        done = False
        t = 0
        while not done and t < ep_max_len:
            last_obs = obs
            env.render()
            time.sleep(0.02)
            action = expert_model.policy.act([obs], expert_model.sess)[0]
            obs, reward, done, info = env.step(action)
            t += 1
        time.sleep(1)

def visualize_irl_policy(env_name, irl_dir, irl_name, irl_algo=AIRL, ep_max_len=100, n_runs=1):
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    irl_model = irl_algo(irl_name, env_fn, None, None, None, checkpoint='{}/{}_model'.format(irl_dir, irl_name))
    env = gym.make(env_name)
    for n in range(n_runs):
        obs = env.reset()
        done = False
        t = 0
        while not done and t < ep_max_len:
            env.render()
            time.sleep(0.02)
            action = irl_model.policy.act([obs], irl_model.sess)[0]
            obs, reward, done, info = env.step(action)
            t += 1
        time.sleep(1)

# works only for pointmaze atm
def visualize_reward(env_name, irl_dir, irl_name, irl_algo=AIRL):
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    irl_model = irl_algo(irl_name, env_fn, None, None, None, checkpoint='{}/{}_model'.format(irl_dir, irl_name))

    rewards = np.zeros((100, 100))
    for i, x in zip(np.arange(100), np.linspace(-0.1, 0.6, 100)):
        for j, y in zip(np.arange(100), np.linspace(-0.1, 0.6, 100)):
            rewards[i, j] = irl_model.discriminator.reward(np.array([[x, y, 0]]), irl_model.sess)

    print('scale:', np.min(rewards), '(black) to', np.max(rewards), '(white)')
    rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
    plt.imshow(rewards, cmap='gray', origin='lower')
    plt.show()

if __name__ == '__main__':
    # target_dict = {
    #     0: np.array([1, 0]),
    #     1: np.array([0, 1]),
    #     2: np.array([-1, 0]),
    #     3: np.array([0, -1]),
    # }
    # for i in range(4):
    #     for j in range(4):
    #         print(i, j)
    #         env = gym.make('PointMass-v{}{}'.format(i, j))
    #         obs, actions, next_obs = [], [], []
    #         for _ in range(50):
    #             rollout_obs, rollout_next_obs, rollout_actions = [], [], []
    #             rollout_obs.append(env.reset())
    #             for _ in range(50):
    #                 action = 4*(target_dict[i]-rollout_obs[-1][:2]) + np.random.normal(size=2)
    #                 ep_obs, reward, done, info = env.step(action)
    #                 rollout_obs.append(ep_obs)
    #                 rollout_next_obs.append(ep_obs)
    #                 rollout_actions.append(action)
    #             for _ in range(50):
    #                 action = 4*(target_dict[j]-rollout_obs[-1][:2]) + np.random.normal(size=2)
    #                 ep_obs, reward, done, info = env.step(action)
    #                 rollout_obs.append(ep_obs)
    #                 rollout_next_obs.append(ep_obs)
    #                 rollout_actions.append(action)
    #             rollout_obs = rollout_obs[:-1]
    #             obs.append(rollout_obs)
    #             next_obs.append(rollout_next_obs)
    #             actions.append(rollout_actions)
    #         obs, next_obs, actions = np.array(obs), np.array(next_obs), np.array(actions)
    #         pickle.dump({'expert_obs': obs, 'expert_next_obs': next_obs, 'expert_actions': actions}, open('data/pointmass/expert{}{}.pkl'.format(i, j), 'wb'))

    # for i in range(4):
    #     for j in range(4):
    #         print('Training', i, j)
#             train_expert(n_iters=200, save_dir='data/pointmass', name='expert-{}'.format(i), env_name='PointMass-v{}'.format(i), use_checkpoint=False, timesteps_per_rollout=1000, ep_max_len=250, demo_timesteps=1e4)
#             visualize_expert(env_name='PointMass-v{}'.format(i), expert_dir='data/pointmass', expert_name='expert-{}'.format(i))

    # train_irl(n_iters=1000, save_dir='data/ant', name='irl', expert_name='expert', env_name='CustomAnt-v0', use_checkpoint=True)
    # visualize_irl_policy(env_name='CustomAnt-v0', irl_dir='data/ant', irl_name='irl')
    # visualize_reward(env_name='PointMazeRight-v0', irl_dir='data/pointmaze', irl_name='irl')
    #
    # train_expert(n_iters=1000, save_dir='data/ant', name='transfer_expert', env_name='DisabledAnt-v0', demo_timesteps=2500, use_checkpoint=True)
    # visualize_expert(env_name='DisabledAnt-v0', expert_dir='data/ant', expert_name='transfer_expert', ep_max_len=250)
