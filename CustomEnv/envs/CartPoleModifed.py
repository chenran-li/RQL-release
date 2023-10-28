import gym
import numpy as np


class CartPoleModifed(gym.Env):

    def __init__(self, render_mode=None):
        self.inner_env = gym.make("CartPole-v1")

        self.observation_space = self.inner_env.observation_space
        
        self.action_space = self.inner_env.action_space

        self.render_mode = render_mode

    def reset(self):
        return self.inner_env.reset()
    
    def step(self, action):

        observation, reward, terminated, info = self.inner_env.step(action)
        
        reward = reward - 10*(np.abs(observation[2])/0.2095)

        return observation, reward, terminated, info

    def render(self, mode, **kwargs):

        return self.inner_env.render(mode, **kwargs)

    def close(self):
        return self.inner_env.close()

class CartPoleModifedMoreCenterAll(gym.Env):

    def __init__(self, render_mode=None):
        self.inner_env = gym.make("CartPole-v1")

        self.observation_space = self.inner_env.observation_space
        
        self.action_space = self.inner_env.action_space

        self.render_mode = render_mode

        self.basic_reward = 0.0

        self.added_reward = 0.0

    def reset(self):
        return self.inner_env.reset()
    
    def step(self, action):

        observation, reward, terminated, info = self.inner_env.step(action)

        self.basic_reward = reward - 10*(np.abs(observation[2])/0.2095)

        self.added_reward = - (np.abs(observation[0])/2.4)

        reward = reward - (np.abs(observation[0])/2.4) - 10*(np.abs(observation[2])/0.2095)

        return observation, reward, terminated, info

    def render(self, mode, **kwargs):

        return self.inner_env.render(mode, **kwargs)

    def close(self):
        return self.inner_env.close()

class CartPoleModifedMoreCenter(gym.Env):

    def __init__(self, render_mode=None):
        self.inner_env = gym.make("CartPole-v1")

        self.observation_space = self.inner_env.observation_space
        
        self.action_space = self.inner_env.action_space

        self.render_mode = render_mode

    def reset(self):
        return self.inner_env.reset()
    
    def step(self, action):

        observation, reward, terminated, info = self.inner_env.step(action)
        reward =  - (np.abs(observation[0])/2.4)
        return observation, reward, terminated, info

    def render(self, mode, **kwargs):

        return self.inner_env.render(mode, **kwargs)

    def close(self):
        return self.inner_env.close()