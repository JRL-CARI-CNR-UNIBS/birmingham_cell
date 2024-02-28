#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import birmingham_envs

from stable_baselines3 import TD3
import rospkg

trees_path = ['/home/battery/michele_projects/personal_ws/src/birmingham_cell/birmingham_cell_tests/config/trees']
epoch_number = 10
max_epoch_steps = 200
learning_start_steps = 100
train_freq = 1

total_timesteps = max_epoch_steps * epoch_number

env = gym.make('ConnectionEnv-v0',action_type='increment_value', max_episode_steps=max_epoch_steps, trees_path=trees_path)
model = TD3("MlpPolicy", env, verbose=1, learning_starts=learning_start_steps, train_freq=train_freq)
vec_env = model.get_env()

rospack = rospkg.RosPack()
path = rospack.get_path('birmingham_cell_tests')
model = TD3.load(path + "/data/td3_model")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")