#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import birmingham_envs

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import rospkg

epoch_number = 50
max_epoch_steps = 200
learning_start_steps = 100
train_freq = 1

total_timesteps = max_epoch_steps * epoch_number

env = gym.make('ConnectionEnv-v0',
               action_type='increment_value', 
               max_episode_steps=max_epoch_steps, 
               # debug_mode=True)
               debug_mode=False)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", 
            env, 
            verbose=1, 
            action_noise=action_noise,
            learning_starts=learning_start_steps, 
            train_freq=train_freq)
model.learn(total_timesteps=total_timesteps, log_interval=1)

rospack = rospkg.RosPack()
path = rospack.get_path('birmingham_cell_tests')

model.save(path + "/data/td3_model_2")


