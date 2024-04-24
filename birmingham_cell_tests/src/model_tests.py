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
import copy

if __name__ == '__main__':

    params_path = sys.argv[1]

    rospack = rospkg.RosPack()
    path = rospack.get_path('birmingham_cell_tests')
    file_path = path + '/' + params_path

    with open(file_path) as file:
        params = yaml.safe_load(file)

    if 'env_type' in params:
        env_type = params['env_type']
    else:
        print('env_type is empty')
        exit(0)
    if 'model_path' in params:
        model_path = params['model_path']
    else:
        print('model_path is empty')
        exit(0)
    if 'model_name' in params:
        model_name = params['model_name']
    else:
        print('model_name is empty')
        exit(0)
    if 'max_epoch_steps' in params:
        max_epoch_steps = params['max_epoch_steps']
    else:
        max_epoch_steps = 25

# env = gym.make('RandomRealFakeEnv-v0', 
#                 action_type='increment_value', 
#                 epoch_len = max_epoch_steps,
#                 max_episode_steps=max_epoch_steps)

    if env_type == 'connection':
        env = gym.make('ConnectionEnv-v0', 
                       action_type='increment_value', 
                       epoch_len = max_epoch_steps,
                       max_episode_steps=max_epoch_steps)
    elif env_type == 'realistic_fake':
        env = gym.make('RealisticFakeEnv-v0', 
                       action_type='increment_value', 
                       epoch_len = max_epoch_steps,
                       max_episode_steps=max_epoch_steps)
    else:
        print('Env_type not exist')
        exit(1)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MlpPolicy", 
                env, 
                verbose=1,
                action_noise=action_noise,
                )

    model = TD3.load("/home/gauss/projects/personal_ws/src/birmingham_cell/birmingham_cell_tests/model/test")

    obs, info = env.reset()
    steps = 0
    success = False

    action = np.array([0,0,0,0,0,0])
    env.step(action)
    
    old_param = np.array(copy.copy(obs[0:6]))
    while (not success) and (steps < max_epoch_steps):
    # while (steps < max_epoch_steps):
        action, _states = model.predict(obs)
        print('observation')
        print(obs)
        print('action')
        print(action)
        print(' ')
        old_param = np.array(copy.copy(obs[0:6]))
        obs, reward, success, truncated, info = env.step(action)
        new_param = np.array(copy.copy(obs[0:6]))
        diff = (new_param - old_param) * 100
        print('calculated action')
        print(diff)
        steps += 1  
np.ndarray.tolist