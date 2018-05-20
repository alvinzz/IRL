import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from IRL.envs.dynamic_mjc.mjc_models import pointmass

# target should be in [0, 1, 2, 3]
class PointMass(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, targets=[0], episode_length=40):
        utils.EzPickle.__init__(self)

        self.max_episode_length = episode_length
        self.targets = targets

        self.episode_length = 0

        model = pointmass(targets)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, frame_skip=5)

    def step(self, a):
        target_num = (self.episode_length * len(self.targets)) // self.max_episode_length

        vec_dist = self.get_body_com("particle") - self.get_body_com("target_{}".format(self.targets[target_num]))
        reward_dist = -np.linalg.norm(vec_dist)**2  # particle to target
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + 0.000 * reward_ctrl

        # targ_v = self.get_body_com("target_{}".format(self.targets[target_num]))
        # reward = -np.linalg.norm(targ_v - self.sim.data.get_body_xvelp("particle"), ord=1)

        self.do_simulation(a, self.frame_skip)
        self.episode_length += 1
        ob = self._get_obs()
        done = self.episode_length >= self.max_episode_length
        return ob, reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel #+ self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        self.episode_length = 0
        return self._get_obs()

    def _get_obs(self):
        target_num = (self.episode_length * len(self.targets)) // self.max_episode_length
        return np.concatenate([
            self.get_body_com("particle"),
            [self.episode_length]
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
