import gym
from gym.envs.registration import register

from rl_zoo3.wrappers import MaskVelocityWrapper

try:
    import pybullet_envs  # pytype: disable=import-error
except ImportError:
    pybullet_envs = None

try:
    import highway_env  # pytype: disable=import-error
except ImportError:
    highway_env = None

try:
    import neck_rl  # pytype: disable=import-error
except ImportError:
    neck_rl = None

try:
    import mocca_envs  # pytype: disable=import-error
except ImportError:
    mocca_envs = None

try:
    import custom_envs  # pytype: disable=import-error
except ImportError:
    custom_envs = None

try:
    import gym_donkeycar  # pytype: disable=import-error
except ImportError:
    gym_donkeycar = None

try:
    import panda_gym  # pytype: disable=import-error
except ImportError:
    panda_gym = None

try:
    import rocket_lander_gym  # pytype: disable=import-error
except ImportError:
    rocket_lander_gym = None


# Register no vel envs
def create_no_vel_env(env_id: str):
    def make_env():
        env = gym.make(env_id)
        env = MaskVelocityWrapper(env)
        return env

    return make_env


for env_id in MaskVelocityWrapper.velocity_indices.keys():
    name, version = env_id.split("-v")
    register(
        id=f"{name}NoVel-v{version}",
        entry_point=create_no_vel_env(env_id),
    )

register(
    id='SimplePuzzle-v0',
    entry_point='CustomEnv.envs.SimplePuzzle:SimplePuzzle',
    max_episode_steps=300,
)

register(
    id='SimplePuzzleNoObs-v0',
    entry_point='CustomEnv.envs.SimplePuzzleNoObs:SimplePuzzleNoObs',
    max_episode_steps=300,
)

register(
    id='SimplePuzzleNoG-v0',
    entry_point='CustomEnv.envs.SimplePuzzleNoG:SimplePuzzleNoG',
    max_episode_steps=300,
)

register(
    id='CartPole-modifed-v1',
    entry_point='CustomEnv.envs.CartPoleModifed:CartPoleModifed',
    max_episode_steps=500,
)

register(
    id='CartPole-modifed-morecenterall-v1',
    entry_point='CustomEnv.envs.CartPoleModifed:CartPoleModifedMoreCenterAll',
    max_episode_steps=500,
)

register(
    id='CartPole-modifed-morecenter-v1',
    entry_point='CustomEnv.envs.CartPoleModifed:CartPoleModifedMoreCenter',
    max_episode_steps=500,
)

register(
    id='MountainCarContinuous-modifed-v0',
    entry_point='CustomEnv.envs.MountainCarContinuousModifed:MountainCarContinuousModifed',
    max_episode_steps=999,
)

register(
    id='MountainCarContinuous-modifed-lessleftall-v0',
    entry_point='CustomEnv.envs.MountainCarContinuousModifed:MountainCarContinuousModifedLessleftActionAll',
    max_episode_steps=999,
)

register(
    id='MountainCarContinuous-modifed-lessleft-v0',
    entry_point='CustomEnv.envs.MountainCarContinuousModifed:MountainCarContinuousModifedLessleftAction',
    max_episode_steps=999,
)