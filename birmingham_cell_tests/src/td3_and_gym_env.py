#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import birmingham_envs

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import rospkg
import os

epoch_number = 1
max_epoch_steps = 30
learning_start_steps = 20
train_freq = 1
start_epoch_number = 0
learning_rate = 0.001
total_epoch = 500
model_name = 'td3_new_model' + str(max_epoch_steps) + '_' + str(learning_rate)
data_name = 'td3_new_tests' + str(max_epoch_steps) + '_' + str(learning_rate)

total_timesteps = max_epoch_steps * epoch_number

env = gym.make('ConnectionEnv-v0',
               action_type='increment_value', 
               max_episode_steps=max_epoch_steps, 
               data_file_name=data_name,
               start_epoch_number=start_epoch_number,
               debug_mode=True)
            #    debug_mode=False)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", 
            env, 
            verbose=1, 
            action_noise=action_noise,
            learning_starts=learning_start_steps, 
            train_freq=train_freq)

rospack = rospkg.RosPack()
path = rospack.get_path('birmingham_cell_tests')
file_path = path + "/data/" + model_name
if os.path.isfile(file_path):
    model.load(file_path)
else:
    if not (start_epoch_number == 0):
        print('There is a problem, you set start_epoch_number equal to ' + str(start_epoch_number) + ' but the model file does not exist')
    else:
        print('epoch 1')
        model.learn(total_timesteps=total_timesteps, log_interval=1)
        model.save(path + "/data/" + model_name)
        model.learning_start_steps = 0 

for i in range(start_epoch_number, total_epoch):
    if (start_epoch_number == 0):
        print('epoch ' + str(i+2))
    else:
        print('epoch ' + str(i+1))
    model.learn(total_timesteps=total_timesteps, log_interval=1)
    model.save(path + "/data/" + model_name)
