import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from rllab.misc import logger

from IRL.envs.dynamic_mjc.mjc_models import pointmass

# target should be in [0, 1, 2, 3]
class PointMass(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, target=0, episode_length=25):
        utils.EzPickle.__init__(self)
        self.max_episode_length = episode_length
        self.target = target

        self.episode_length = 0

        model = pointmass(target)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, frame_skip=5)

    def step(self, a):
        vec_dist = self.get_body_com("particle") - self.get_body_com("target_{}".format(self.target))

        reward_dist = - np.linalg.norm(vec_dist)  # particle to target
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + 0.000 * reward_ctrl

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        self.episode_length += 1
        done = self.episode_length >= self.max_episode_length
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos
        self.episode_length = 0
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        self.episode_length = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.get_body_com("particle"),
            # self.get_body_com("target"),
        ])

    def plot_trajs(self, *args, **kwargs):
        pass

    def log_diagnostics(self, paths):
        rew_dist = np.array([traj['env_infos']['reward_dist'] for traj in paths])
        rew_ctrl = np.array([traj['env_infos']['reward_ctrl'] for traj in paths])

        logger.record_tabular('AvgObjectToGoalDist', -np.mean(rew_dist.mean()))
        logger.record_tabular('AvgControlCost', -np.mean(rew_ctrl.mean()))
        logger.record_tabular('AvgMinToGoalDist', np.mean(np.min(-rew_dist, axis=1)))
