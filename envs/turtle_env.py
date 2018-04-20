import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import pdb

class TurtleEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.viewer = None
        self.max_linear_vel = 10
        self.max_angular_vel = 5
        self.dt = 0.05

        self.height = 100.0
        self.width = 100.0
        high = np.array([self.max_linear_vel, self.max_angular_vel])

        # Range of linear and angular velocities that the turtlebot accepts as input.
        self.action_space = spaces.Box(low=-high, high=high)

        # Range of coordinates and orientation angles which define the state space.
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([self.width, self.height, 2*np.pi]))

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        # Current coordinates and orientation angle.
        x, y, th = self.state # th := theta

        dt = self.dt
        v = np.clip(u[0], -self.max_linear_vel, self.max_linear_vel)
        th_dot = np.clip(u[1], -self.max_angular_vel, self.max_angular_vel)

        self.last_u = [v, th_dot] # for rendering

        dx  = v * np.cos(th) * dt
        dy  = v * np.sin(th) * dt
        dth = th_dot * dt

        new_x  = np.clip(x + dx, 0, self.width)
        new_y  = np.clip(y + dy, 0, self.height)
        new_th = th + dth
        self.state = np.array([new_x, new_y, new_th])

        costs = 0 # for now!
        return self._get_obs(), -costs, False, {}

    def reset(self):
        # Start in the middle of the board with an angle of 0
        self.state = np.array([self.width / 2, self.height / 2, 0])
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            # self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            self.viewer.set_bounds(0, self.width, 0, self.height)
            turtle = rendering.make_circle(3)
            turtle.set_color(0, 0, 0)
            self.turtle_transform = rendering.Transform()
            turtle.add_attr(self.turtle_transform)
            self.viewer.add_geom(turtle)

        self.turtle_transform.set_translation(self.state[0], self.state[1])
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()


env = TurtleEnv()
env.reset()
for _ in range(10000):
    env.render()
    env.step(env.action_space.sample())