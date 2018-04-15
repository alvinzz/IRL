import gym
from algos import AIRL
from rewards import env_reward_fn

if __name__ == '__main__':
    env_fn = lambda: gym.make('Pendulum-v0')
    model = AIRL(env_fn, 3, 1, env_reward_fn, [], [], [])
    model.train(200, 10, 10)
