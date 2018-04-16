import gym
from algos import AIRL
from rewards import env_reward_fn, make_AIRL_reward_fn
from rollouts import collect_and_process_rollouts

def collect_pendulum_data():
    env_fn = lambda: gym.make('Pendulum-v0')
    model = AIRL('expert', env_fn, 3, 1, None, None, None)
    model.train(200, 1000, 100, reward_fn=env_reward_fn)
    expert_obs, expert_next_obs, expert_actions, _, _, _, _ = collect_and_process_rollouts(env_fn, model.policy, env_reward_fn, model.sess, 10000, 100)
    return expert_obs, expert_next_obs, expert_actions

def train_pendulum_irl(expert_obs, expert_next_obs, expert_actions):
    env_fn = lambda: gym.make('Pendulum-v0')
    model = AIRL('airl', env_fn, 3, 1, expert_obs, expert_next_obs, expert_actions)
    model.train(200, 1000, 100, reward_fn=make_AIRL_reward_fn(model.discriminator, model.sess))
    return model

if __name__ == '__main__':
    expert_obs, expert_next_obs, expert_actions = collect_pendulum_data()
    model = train_pendulum_irl(expert_obs, expert_next_obs, expert_actions)
