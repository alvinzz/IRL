import gym
from algos import AIRL
from rewards import env_reward_fn, make_AIRL_reward_fn
from rollouts import collect_and_process_rollouts
import tensorflow as tf
import numpy as np
import pickle

def IRL_launcher(
    n_iters, save_dir, expert_dir,
    env_name, timesteps_per_rollout=2000, ep_max_len=1000,
    train_expert=False, expert_checkpoint=None, expert_timesteps=1e5, # number of timesteps in expert demonstration data
    irl_algo=AIRL, irl_make_reward_fn=make_AIRL_reward_fn, irl_checkpoint=None
):
    env_fn = lambda: gym.make(env_name)

    if train_expert:
        expert_model = irl_algo('expert', env_fn, None, None, None, checkpoint=expert_checkpoint)
        print('\nTraining expert...')
        expert_model.train(n_iters, timesteps_per_rollout, ep_max_len, reward_fn=env_reward_fn)
        print('\nCollecting expert trajectories...')
        expert_obs, expert_next_obs, expert_actions, _, _, _, _ = collect_and_process_rollouts(env_fn, expert_model.policy, env_reward_fn, expert_model.sess, expert_timesteps, ep_max_len)
        pickle.dump({'expert_obs': expert_obs, 'expert_next_obs': expert_next_obs, 'expert_actions': expert_actions}, open('{}/expert.pkl'.format(save_dir), 'wb'))
        expert_model.saver.save(expert_model.sess, '{}/expert_model'.format(save_dir))

    data = pickle.load(open('{}/expert.pkl'.format(expert_dir), 'rb'))
    expert_obs, expert_next_obs, expert_actions = data['expert_obs'], data['expert_next_obs'], data['expert_actions']

    print('\nTraining IRL...')
    irl_model = irl_algo('irl', env_fn, expert_obs, expert_next_obs, expert_actions, checkpoint=irl_checkpoint)
    irl_model.train(n_iters, timesteps_per_rollout, ep_max_len, reward_fn=irl_make_reward_fn(irl_model.discriminator, irl_model.sess))

    # evaluate and save model
    print('\nEvaluating policy on original task...')
    collect_and_process_rollouts(env_fn, irl_model.policy, env_reward_fn, irl_model.sess, expert_timesteps, ep_max_len)
    irl_model.saver.save(irl_model.sess, '{}/irl_model'.format(save_dir))

    return irl_model

if __name__ == '__main__':
    IRL_launcher(n_iters=500, save_dir='data/halfcheetah', expert_dir='data/halfcheetah', env_name='HalfCheetah-v2')
