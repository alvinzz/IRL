import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import pdb
import tty, termios, sys

def getch():
    """getch() -> key character

    Read a single keypress from stdin and return the resulting character.
    Nothing is echoed to the console. This call will block if a keypress
    is not already available, but will not wait for Enter to be pressed.

    If the pressed key was a modifier key, nothing will be detected; if
    it were a special function key, it may return the first character of
    of an escape sequence, leaving additional characters in the buffer.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

class TurtleEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.viewer = None
        self.max_linear_vel = 0.1
        self.max_angular_vel = 5
        self.dt = 0.05
        self.box = np.random.rand(2)
        self.box_angle = np.pi/4 * (np.random.rand() - 0.5)
        self.target = np.random.rand(2)

        self.height = 1
        self.width = 1
        high = np.array([self.max_linear_vel, self.max_angular_vel])

        # Range of linear and angular velocities that the turtlebot accepts as input.
        self.action_space = spaces.Box(low=-high, high=high)

        # Range of coordinates and orientation angles which define the state space.
        # distance to box_face, angle between current orientation and orientation which points at box_face,
        # distance to box, angle between current orientation and orientation which points at box,
        # distance to target, angle between current orientation and orientation which points at target,
        max_dist = np.sqrt(self.width**2 + self.height**2)
        self.observation_space = spaces.Box(low=np.array([0, -np.pi, 0, -np.pi, 0, -np.pi]), high=np.array([max_dist, np.pi, max_dist, np.pi, max_dist, np.pi]))

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

        dx  = v * np.cos(th) * dt
        dy  = v * np.sin(th) * dt
        dth = th_dot * dt

        new_x  = x + dx
        new_y  = y + dy
        new_th = th + dth
        while new_th < -np.pi:
            new_th += 2*np.pi
        while new_th >= np.pi:
            new_th -= 2*np.pi
        self.state = np.array([new_x, new_y, new_th])

        return self._get_obs(), 0, False, {}

    def reset(self):
        # Start in the middle of the board with an angle of 0
        # self.state = np.array([self.width / 2, self.height / 2, 0])
        self.box = np.zeros(2)
        self.state = np.zeros(3)
        self.target = np.zeros(2)
        while np.linalg.norm(self.box - self.state[:2]) <= 0.5 or np.linalg.norm(self.box - self.target) <= 0.5:
            self.box = np.random.rand(2)
            self.box_angle = np.pi/4 * (np.random.rand() - 0.5)
            self.target = np.random.rand(2)
            self.state = np.array([np.random.rand(), np.random.rand(), 2*np.pi*(np.random.rand()-0.5)])
            self.start_box_angle = np.arctan2(self.box[1] - self.state[1], self.box[0] - self.state[0])
        return self._get_obs()

    def _get_obs(self):
        return np.array([
            np.linalg.norm(self.state[:2] - self._get_box_face_coords()),
            self._get_box_face_angle(),
            np.linalg.norm(self.state[:2] - self.box),
            self._get_box_angle(),
            np.linalg.norm(self.state[:2] - self.target),
            self._get_target_angle()
        ])

    def _get_box_face_coords(self, box_face_dist=0.20):
        angle = self.start_box_angle + np.pi + self.box_angle
        offset = box_face_dist * np.array([np.cos(angle), np.sin(angle)])
        return self.box + offset

    def _get_box_angle(self):
        return self._get_angle(self.box)

    def _get_box_face_angle(self):
        return self._get_angle(self._get_box_face_coords())

    def _get_target_angle(self):
        return self._get_angle(self.target)

    def _get_angle(self, other):
        angle = np.arctan2(other[1] - self.state[1], other[0] - self.state[0]) - self.state[2]
        while angle < -np.pi:
            angle += 2*np.pi
        while angle >= np.pi:
            angle -= 2*np.pi
        return angle

    def render(self, mode='human'):
        turtle_radius = 0.065
        box_radius = 0.03
        arm_length = 0.12
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(0, self.width, 0, self.height)

            turtle = rendering.make_circle(turtle_radius)
            turtle.set_color(0, 0, 0)
            self.turtle_transform = rendering.Transform()
            turtle.add_attr(self.turtle_transform)
            self.viewer.add_geom(turtle)

            left = rendering.make_capsule(arm_length, .005)
            left.set_color(.8, .3, .3)
            self.left_arm = rendering.Transform()
            left.add_attr(self.left_arm)
            self.viewer.add_geom(left)

            right = rendering.make_capsule(arm_length, .005)
            right.set_color(.8, .3, .3)
            self.right_arm = rendering.Transform()
            right.add_attr(self.right_arm)
            self.viewer.add_geom(right)

            box = rendering.make_circle(box_radius)
            box.set_color(0, 16, 0)
            self.box_transform = rendering.Transform()
            box.add_attr(self.box_transform)
            self.viewer.add_geom(box)

            box_line = rendering.make_capsule(box_radius, .005)
            box_line.set_color(0, 0, 0)
            self.box_line = rendering.Transform()
            box_line.add_attr(self.box_line)
            self.viewer.add_geom(box_line)

            target = rendering.make_circle(0.025)
            target.set_color(16, 0, 0)
            self.target_transform = rendering.Transform()
            target.add_attr(self.target_transform)
            self.viewer.add_geom(target)

        self.turtle_transform.set_translation(self.state[0], self.state[1])

        # Left and right arm orientation.
        th = self.state[2]
        rotation = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])

        left_offset = np.dot(rotation, np.array([[0], [turtle_radius*2/3]]))
        self.left_arm.set_translation(self.state[0] + left_offset[0][0], self.state[1] + left_offset[1][0])
        self.left_arm.set_rotation(self.state[2])

        right_offset = np.dot(rotation, np.array([[0], [-turtle_radius*2/3]]))
        self.right_arm.set_translation(self.state[0] + right_offset[0][0], self.state[1] + right_offset[1][0])
        self.right_arm.set_rotation(self.state[2])

        # Handle collision with the turtlebot.
        # If the distance between the turtlebot and the box is less than the sum of the radii...
        if np.linalg.norm(self.state[0:2] - self.box) < turtle_radius + box_radius:
            overlap = np.linalg.norm(self.state[0:2] - self.box) - (turtle_radius + box_radius)
            offset = overlap * (self.state[0:2] - self.box) / np.linalg.norm(self.state[0:2] - self.box)
            self.box += offset

        # Left arm collisions
        s1 = np.array([self.state[0] + left_offset[0][0], self.state[1] + left_offset[1][0]])
        s2 = s1 + np.dot(rotation, np.array([[arm_length], [0]])).T[0]

        t = np.dot(self.box - s1, s2 - s1) / (np.linalg.norm(s2 - s1) ** 2)
        t = min(max(t, 0), 1)
        min_dist_vec = self.box - (s1 + t*(s2 - s1))
        left_dist = np.linalg.norm(min_dist_vec)
        if left_dist < box_radius:
            overlap = box_radius - left_dist
            self.box += overlap * min_dist_vec / np.linalg.norm(min_dist_vec)

        # Right arm collisions
        s1 = np.array([self.state[0] + right_offset[0][0], self.state[1] + right_offset[1][0]])
        s2 = s1 + np.dot(rotation, np.array([[arm_length], [0]])).T[0]

        t = np.dot(self.box - s1, s2 - s1) / (np.linalg.norm(s2 - s1) ** 2)
        t = min(max(t, 0), 1)
        min_dist_vec = self.box - (s1 + t*(s2 - s1))
        right_dist = np.linalg.norm(min_dist_vec)
        if right_dist < box_radius:
            overlap = box_radius - right_dist
            self.box += overlap * min_dist_vec / np.linalg.norm(min_dist_vec)

        self.box_transform.set_translation(self.box[0], self.box[1])
        self.box_line.set_translation(self.box[0], self.box[1])
        self.box_line.set_rotation(self.start_box_angle + np.pi + self.box_angle)

        self.target_transform.set_translation(self.target[0], self.target[1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

if __name__ == '__main__':
    frequencies = np.zeros(4)
    env = TurtleEnv()
    obs = []
    actions = []
    next_obs = []
    for iter_ in range(100):
        print(iter_)
        ob = env.reset()
        obs.append(ob)
        # print("Press 'q' to exit...\n")
        while ob[0] > 0.01:
            # env.render()
            action = (np.clip(ob[0], 0.01, 0.1), ob[1])
            ob, _, _, _ = env.step(action)
            obs.append(ob)
            actions.append(action)
            next_obs.append(ob)
            frequencies[0] += 1
        while np.abs(ob[3]) > 0.01:
            # env.render()
            action = (0, ob[3])
            ob, _, _, _ = env.step(action)
            obs.append(ob)
            actions.append(action)
            next_obs.append(ob)
            frequencies[1] += 1
        while ob[2] > 0.095:
            # env.render()
            action = (np.clip(ob[2]-0.095, 0.01, 0.1), 0)
            ob, _, _, _ = env.step(action)
            obs.append(ob)
            actions.append(action)
            next_obs.append(ob)
            frequencies[2] += 1
        while ob[4] > 0.095:
            # env.render()
            action = (np.clip(ob[4]-0.095, 0.01, 0.1), ob[5])
            ob, _, _, _ = env.step(action)
            obs.append(ob)
            actions.append(action)
            next_obs.append(ob)
            frequencies[3] += 1
        obs = obs[:-1]
    print(frequencies/np.sum(frequencies))
    # obs, actions, next_obs = np.array(obs), np.array(actions), np.array(next_obs)
    # import pickle
    # pickle.dump({
    #     'expert_obs': obs,
    #     'expert_actions': actions,
    #     'expert_next_obs': next_obs
    # },
    # open('../data/turtle/intention_expert.pkl', 'wb'))

    # command = getch()
    # if command == 'q':
    #     break
    # if command == 'w':
    #     env.step((.1, 0))
    # elif command == 's':
    #     env.step((-.1, 0))
    # elif command == 'a':
    #     env.step((0, 3))
    # elif command == 'd':
    #     env.step((0, -3))
