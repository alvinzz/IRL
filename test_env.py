from envs import *
import gym

from gym.envs import register

register(id='PointMazeRight-v0', entry_point='IRL.envs.point_maze_env:PointMazeEnv',
          kwargs={'sparse_reward': False, 'direction': 1})
env = gym.make('PointMazeRight-v0')
env.reset()
while True:
    env.render()
    env.step(env.action_space.sample())
