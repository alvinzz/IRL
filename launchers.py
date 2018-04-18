import gym
from algos import AIRL
from rewards import env_reward_fn, make_AIRL_reward_fn
from rollouts import collect_and_process_rollouts
import tensorflow as tf
import numpy as np
import pickle

def collect_hc_data(save_dir='data/halfcheetah', checkpoint=None):
    env_fn = lambda: gym.make('HalfCheetah-v2')
    model = AIRL('expert', env_fn, None, None, None, checkpoint=checkpoint)
    try:
        model.train(500, 2000, 1000, reward_fn=env_reward_fn)
    except Exception as e:
        model.saver.save(model.sess, '{}/failed_expert_model'.format(save_dir))

    # get expert rollouts and save model
    expert_obs, expert_next_obs, expert_actions, _, _, _, _ = collect_and_process_rollouts(env_fn, model.policy, env_reward_fn, model.sess, 100000, 1000)
    pickle.dump({'expert_obs': expert_obs, 'expert_next_obs': expert_next_obs, 'expert_actions': expert_actions}, open('{}/expert.pkl'.format(save_dir), 'wb'))
    model.saver.save(model.sess, '{}/expert_model'.format(save_dir))

    return expert_obs, expert_next_obs, expert_actions

def train_hc_irl(expert_obs, expert_next_obs, expert_actions, save_dir='data/halfcheetah', checkpoint=None):
    env_fn = lambda: gym.make('HalfCheetah-v2')
    model = AIRL('airl', env_fn, expert_obs, expert_next_obs, expert_actions, checkpoint=checkpoint)
    try:
        model.train(500, 2000, 1000, reward_fn=make_AIRL_reward_fn(model.discriminator, model.sess))
    except Exception as e:
        model.saver.save(model.sess, '{}/failed_irl_model'.format(save_dir))

    # evaluate and save model
    collect_and_process_rollouts(env_fn, model.policy, env_reward_fn, model.sess, 100000, 1000)
    model.saver.save(model.sess, '{}/irl_model'.format(save_dir))

    return model

if __name__ == '__main__':
    # expert_obs, expert_next_obs, expert_actions = collect_hc_data()

    data = pickle.load(open('data/halfcheetah/expert.pkl', 'rb'))
    expert_obs, expert_next_obs, expert_actions = data['expert_obs'], data['expert_next_obs'], data['expert_actions']
    model = train_hc_irl(expert_obs, expert_next_obs, expert_actions, checkpoint='data/halfcheetah/irl_model')
