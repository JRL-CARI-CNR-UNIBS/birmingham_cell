#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import birmingham_envs

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


epoch_number = 5
max_epoch_steps = 10
learning_start_steps = 5
train_freq = 1

total_timesteps = max_epoch_steps * epoch_number

env = gym.make('ConnectionEnv-v0',action_type='increment_value', max_episode_steps=max_epoch_steps)
model = TD3("MlpPolicy", env, verbose=1, learning_starts=learning_start_steps, train_freq=train_freq)
model.learn(total_timesteps=total_timesteps, log_interval=1)

model.save("/home/gauss/projects/personal_ws/src/birmingham_cell/birmingham_cell_tests/data/td3_model")


