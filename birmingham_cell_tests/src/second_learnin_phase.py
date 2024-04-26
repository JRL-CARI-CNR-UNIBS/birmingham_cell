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
    if 'max_epoch_steps' in params:
        max_epoch_steps = params['max_epoch_steps']
    else:
        max_epoch_steps = 25
    if 'learning_rate' in params:
        learning_rate = params['learning_rate']
    else:
        learning_rate = 0.01
    if 'gamma' in params:
        gamma = params['gamma']
    else:
        gamma = 0.9

    if 'saving_name_space' in params:
        name_space = params['saving_name_space']
    else:
        name_space = 'generic_tests'
    if 'verbose' in params:
        verbose = params['verbose']
    else:
        verbose = 0
             
    if 'debug_mode' in params:
        debug_mode = params['debug_mode']
    else:
        debug_mode = False
    if 'only_pos_success' in params:
        only_pos_success = params['only_pos_success']
    else:
        only_pos_success = True
    if 'obj_pos_error' in params:
        obj_pos_error = params['obj_pos_error']
    else:
        only_pos_success = [0.0,0.0,0.0]
    if 'model_name' in params:
        model_name = params['model_name']
    else:
        model_name = 'test'

    data = datetime.datetime.now()
    test_name = name_space + 'tests'
    
    log_repo_path = path + '/log' 
    models_repo_path = path + '/model/'

    if 'model_save_freq' in params:
        model_save_freq = params['model_save_freq']
    else:
        model_save_freq = max_epoch_steps
        
    if params['env_type'] == 'connection':
        env = gym.make('ConnectionEnv-v0', 
                        action_type='increment_value', 
                        debug_mode=debug_mode,
                        only_pos_success=only_pos_success,
                        epoch_len = max_epoch_steps,
                        max_episode_steps=max_epoch_steps)
    elif params['env_type'] == 'realistic_fake':
        print('In realistic_fake')
        env = gym.make('RealisticFakeEnv-v0', 
                        action_type='increment_value', 
                        epoch_len = max_epoch_steps,
                        max_episode_steps=max_epoch_steps,
                        obj_pos_error=obj_pos_error,
                        )
    elif params['env_type'] == 'random_real_fake':
        print('In random_real_fake')
        env = gym.make('RandomRealFakeEnv-v0', 
                        action_type='increment_value', 
                        debug_mode=debug_mode,
                        only_pos_success=only_pos_success,
                        epoch_len = max_epoch_steps,
                        max_episode_steps=max_epoch_steps)
    elif params['env_type'] == 'generic_real_fake':
        print('In generic_real_fake')
        env = gym.make('GenericRealFakeEnv-v0', 
                        action_type='increment_value', 
                        debug_mode=debug_mode,
                        only_pos_success=only_pos_success,
                        epoch_len = max_epoch_steps,
                        max_episode_steps=max_epoch_steps)
    else:
        print('Env_type not in the possible env list.')
        exit(0)  
          
    model = TD3.load(models_repo_path + model_name)

    print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
    model.set_env(env)

    model_name = params['env_type'] + '/' + str(max_epoch_steps) + '/' + str(learning_rate) + '/' + str(gamma)
    log_name = params['env_type'] + '/' + str(max_epoch_steps) + '/' + str(learning_rate) + '/' + str(gamma)
    model_path = models_repo_path + '/second_' + model_name
    log_path = log_repo_path + '/second_' + log_name

    checkpoint_callback = CheckpointCallback(
        save_freq=model_save_freq,
        save_path=model_path + '/',
        name_prefix=name_space,
    )

    # env.reset()
    
    model.learn(total_timesteps=params['total_timesteps'], 
                log_interval=1, 
                callback=checkpoint_callback,
                )
    
    model.save(model_path)
