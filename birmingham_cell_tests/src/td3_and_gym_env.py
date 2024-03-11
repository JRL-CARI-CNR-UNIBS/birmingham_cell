#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import birmingham_envs

from stable_baselines3 import TD3
# from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

import rospkg
import os

# from gymnasium import NormalizeObservation
# from stable_baselines3.common.vec_env import VecNormalize


test_name = "td3_2_"
epoch_number = 500
max_epoch_steps = 60
learning_start_steps = 2
train_freq = 1
start_epoch_number = 0
learning_rate = 0.001
gamma = 0.99
total_timesteps = max_epoch_steps * epoch_number
model_save_freq = 15

rospack = rospkg.RosPack()
path = rospack.get_path('birmingham_cell_tests')

model_repo_path = path + '/model'
data_repo_path = path + '/data'
log_repo_path = path + '/log'
models_name = test_name + '_models_' + str(gamma) + '_' + str(learning_rate)
model_name = test_name + '_model_' + str(gamma) + '_' + str(learning_rate)
data_name = test_name + '_data_' + str(gamma) + '_' + str(learning_rate)
log_name = test_name + '_log_' + str(gamma) + '_' + str(learning_rate)
model_path = model_repo_path + '/' + model_name
data_path = data_repo_path + '/' + data_name
log_path = log_repo_path + '/' + log_name
models_repo_path = model_repo_path + '/' + models_name

env = gym.make('ConnectionEnv-v0',
               action_type='increment_value', 
               max_episode_steps=max_epoch_steps, 
               data_file_name=data_name,
               start_epoch_number=start_epoch_number,
            #    debug_mode=True)
               debug_mode=False)

# env = gym.NormalizeObservation(env)

checkpoint_callback = CheckpointCallback(
    save_freq=model_save_freq,
    save_path=models_repo_path + '/',
    name_prefix=model_name,
)

# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", 
            env, 
            verbose=1, 
            # action_noise=action_noise,
            learning_rate=learning_rate,
            learning_starts=learning_start_steps, 
            tensorboard_log=log_path,
            train_freq=train_freq,
            gamma=gamma)

model.learn(total_timesteps=total_timesteps, log_interval=1, callback=checkpoint_callback)

model.save(model_path)
