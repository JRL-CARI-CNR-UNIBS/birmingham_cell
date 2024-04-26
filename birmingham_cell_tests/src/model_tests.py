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
    model_path = path + '/' + 'model/'
    
    with open(file_path) as file:
        params = yaml.safe_load(file)

    if 'env_type' in params:
        env_type = params['env_type']
    else:
        print('env_type is empty')
        exit(0)
    if 'model_path' in params:
        model_path = model_path + params['model_path']
    else:
        print('model_path is empty')
        model_path = model_path[:-1]
    if 'model_name' in params:
        model_name = params['model_name']
    else:
        print('model_name is empty')
        exit(0)
    if 'max_epoch_steps' in params:
        max_epoch_steps = params['max_epoch_steps']
    else:
        max_epoch_steps = 25



    if env_type == 'connection':
        if 'obj_type' in params:
            obj_model_name = params['obj_type']
            if obj_model_name == 'can':
                obj_model_height = 0.115
                obj_model_width =  0.06
                obj_model_length = 0.0
            if obj_model_name == 'cylinder':
                obj_model_height = params['obj_height']
                obj_model_width =  params['obj_radius'] * 2
                obj_model_length = 0.0
            if obj_model_name == 'sphere':
                obj_model_height = params['obj_radius'] * 2
                obj_model_width =  0.0
                obj_model_length = 0.0
            if obj_model_name == 'box':
                obj_model_height = params['obj_height']
                obj_model_width =  params['obj_width']
                obj_model_length = params['obj_length']
        env = gym.make('ConnectionEnv-v0', 
                       action_type='increment_value', 
                       epoch_len = max_epoch_steps,
                       max_episode_steps=max_epoch_steps,
                       obj_model_name = obj_model_name,
                       obj_model_height = obj_model_height,
                       obj_model_width =  obj_model_width,
                       obj_model_length = obj_model_length,
                       )
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

    model = TD3.load(model_path + "/" + model_name)

    obs, info = env.reset()
    steps = 0
    success = False

    # action = np.array([0,0,0,0,0,0])
    # env.step(action)
    
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