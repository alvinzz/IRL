import gym
from algos import AIRL
from rewards import env_reward_fn, make_AIRL_reward_fn
from rollouts import collect_and_process_rollouts

import tensorflow as tf
import numpy as np
import pickle

def collect_pendulum_data(save_dir='data/pendulum'):
    env_fn = lambda: gym.make('Pendulum-v0')
    model = AIRL('expert', env_fn, None, None, None)
    model.train(200, 1000, 100, reward_fn=env_reward_fn)

    # get expert rollouts and save model
    expert_obs, expert_next_obs, expert_actions, _, _, _, _ = collect_and_process_rollouts(env_fn, model.policy, env_reward_fn, model.sess, 10000, 100)
    pickle.dump({'expert_obs': expert_obs, 'expert_next_obs': expert_next_obs, 'expert_actions': expert_actions}, open('{}/expert.pkl'.format(save_dir), 'wb'))
    model.saver.save(model.sess, '{}/expert_model'.format(save_dir))

    return expert_obs, expert_next_obs, expert_actions

def train_pendulum_irl(expert_obs, expert_next_obs, expert_actions, save_dir='data/pendulum'):
    env_fn = lambda: gym.make('Pendulum-v0')
    model = AIRL('airl', env_fn, expert_obs, expert_next_obs, expert_actions)
    model.train(200, 1000, 100, reward_fn=make_AIRL_reward_fn(model.discriminator, model.sess))

    # save model
    model.saver.save(model.sess, '{}/irl_model'.format(save_dir))

    return model

def collect_hc_data(save_dir='data/halfcheetah'):
    env_fn = lambda: gym.make('HalfCheetah-v2')
    model = AIRL('expert', env_fn, None, None, None)
    model.train(2000, 5000, 1000, reward_fn=env_reward_fn)

    # get expert rollouts and save model
    expert_obs, expert_next_obs, expert_actions, _, _, _, _ = collect_and_process_rollouts(env_fn, model.policy, env_reward_fn, model.sess, 10000, 100)
    pickle.dump({'expert_obs': expert_obs, 'expert_next_obs': expert_next_obs, 'expert_actions': expert_actions}, open('{}/expert.pkl'.format(save_dir), 'wb'))
    model.saver.save(model.sess, '{}/expert_model'.format(save_dir))

    return expert_obs, expert_next_obs, expert_actions

def train_hc_irl(expert_obs, expert_next_obs, expert_actions, save_dir='data/halfcheetah'):
    env_fn = lambda: gym.make('HalfCheetah-v2')
    model = AIRL('airl', env_fn, expert_obs, expert_next_obs, expert_actions)
    model.train(2000, 5000, 1000, reward_fn=make_AIRL_reward_fn(model.discriminator, model.sess))

    # save model
    model.saver.save(model.sess, '{}/irl_model'.format(save_dir))

    return model

if __name__ == '__main__':
    # expert_obs, expert_next_obs, expert_actions = collect_pendulum_data()
    # model = train_pendulum_irl(expert_obs, expert_next_obs, expert_actions)
    expert_obs, expert_next_obs, expert_actions = collect_hc_data()
    model = train_hc_irl(expert_obs, expert_next_obs, expert_actions)
