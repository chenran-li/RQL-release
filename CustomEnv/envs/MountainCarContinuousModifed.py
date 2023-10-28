import gym
import numpy as np


class MountainCarContinuousModifed(gym.Env):

    def __init__(self, render_mode=None):
        self.inner_env = gym.make("MountainCarContinuous-v0")

        self.observation_space = self.inner_env.observation_space
        
        self.action_space = self.inner_env.action_space

        self.render_mode = render_mode

    def reset(self):
        return self.inner_env.reset()
    
    def step(self, action):

        observation, reward, terminated, info = self.inner_env.step(action)

        return observation, reward, terminated, info

    def render(self, mode, **kwargs):

        return self.inner_env.render(mode, **kwargs)

    def close(self):
        return self.inner_env.close()


class MountainCarContinuousModifedLessleftActionAll(gym.Env):

    def __init__(self, render_mode=None):
        self.inner_env = gym.make("MountainCarContinuous-v0")

        self.observation_space = self.inner_env.observation_space
        
        self.action_space = self.inner_env.action_space

        self.render_mode = render_mode

        self.basic_reward = 0.0

        self.added_reward = 0.0

    def reset(self):
        return self.inner_env.reset()
    
    def step(self, action):

        observation, reward, terminated, info = self.inner_env.step(action)

        self.basic_reward = reward
        self.added_reward = 0.0
        if (action<0):
            reward = reward - 0.1
            self.added_reward = - 0.1

        return observation, reward, terminated, info

    def render(self, mode, **kwargs):

        return self.inner_env.render(mode, **kwargs)

    def close(self):
        return self.inner_env.close()

class MountainCarContinuousModifedLessleftAction(gym.Env):

    def __init__(self, render_mode=None):
        self.inner_env = gym.make("MountainCarContinuous-v0")

        self.observation_space = self.inner_env.observation_space
        
        self.action_space = self.inner_env.action_space

        self.render_mode = render_mode

    def reset(self):
        return self.inner_env.reset()
    
    def step(self, action):

        observation, reward, terminated, info = self.inner_env.step(action)
        reward = 0
        if (action<0):
            reward = reward - 0.1
        return observation, reward, terminated, info

    def render(self, mode, **kwargs):

        return self.inner_env.render(mode, **kwargs)

    def close(self):
        return self.inner_env.close()