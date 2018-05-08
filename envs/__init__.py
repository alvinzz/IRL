import logging
import sys, os
from gym.envs import register

LOGGER = logging.getLogger(__name__)

_REGISTERED = False
def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    IRL_ABS_PATH = "/home/cc/ee106b/sp18/class/ee106b-aax/ros_workspaces/IRL/envs"
    sys.path.append(os.path.abspath(IRL_ABS_PATH))
    # turtlebot
    register(id='Turtle-v0', entry_point='envs.turtle_env:TurtleEnv')
