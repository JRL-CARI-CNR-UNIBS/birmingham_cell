#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import birmingham_envs
import sys

from stable_baselines3 import TD3, SAC, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

import rospkg
import os
import yaml
import datetime

max_epoch_steps = 25

env = gym.make('ConnectionEnv-v0', 
                action_type='increment_value', 
                epoch_len = max_epoch_steps,
                max_episode_steps=max_epoch_steps)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", 
            env, 
            verbose=1,
            action_noise=action_noise,
            )

model = TD3.load("/home/gauss/projects/personal_ws/src/birmingham_cell/birmingham_cell_tests/model/realistic_fake/25/0.01/0.9/realistic_tests__100000_steps")

obs, info = env.reset()
steps = 0
success = False

while (not success) or (steps < max_epoch_steps):
    action, _states = model.predict(obs)
    obs, reward, success, truncated, info = env.step(action)
    steps += 1
