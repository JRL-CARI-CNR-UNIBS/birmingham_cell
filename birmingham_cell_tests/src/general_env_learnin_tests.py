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

    possible_model_type = ['td3','sac','ddpg']

    rospack = rospkg.RosPack()
    path = rospack.get_path('birmingham_cell_tests')
    file_path = path + '/' + params_path

    with open(file_path) as file:
        params = yaml.safe_load(file)

    if 'name_space' in params:
        name_space = params['name_space']
    else:
        name_space = 'learning_tests'
        print('name_space: learning_tests')

    if 'verbose' in params:
        verbose = params['verbose']
    else:
        verbose = 0          
        print('verbose: 0')
    if 'noise_sigma' in params:
        noise_sigma_vec = params['noise_sigma']
        if isinstance(noise_sigma_vec, float):
            noise_sigma_vec = [noise_sigma_vec]
        elif isinstance(noise_sigma_vec, list):
            if not isinstance(noise_sigma_vec[0], float):
                print('noise_sigma_vec is not a list of float')
                exit(1)
        else:
            print('noise_sigma_vec is not a float or a list of float')
            exit(1)
    else:
        noise_sigma_vec = [0.1]
        print('noise_sigma: [0.1]')
    if 'max_epoch_steps' in params:
        max_episode_steps_vec = params['max_epoch_steps']
        if isinstance(max_episode_steps_vec, int):
            max_episode_steps_vec = [max_episode_steps_vec]
        elif isinstance(max_episode_steps_vec, list):
            if not isinstance(max_episode_steps_vec[0], int):
                print('max_epoch_steps_vec is not a list of int')
                exit(1)
        else:
            print('max_epoch_steps_vec is not a int or a list of int')
            exit(1)
    else:
        max_episode_steps_vec = [25]
        print('max_epoch_steps_vec: [25]')
    if 'learning_rate' in params:
        learning_rate_vec = params['learning_rate']
        if isinstance(learning_rate_vec, float):
            learning_rate_vec = [learning_rate_vec]
        elif isinstance(learning_rate_vec, list):
            if not isinstance(learning_rate_vec[0], float):
                print('learning_rate_vec is not a list of float')
                exit(1)
        else:
            print('learning_rate_vec is not a float or a list of float')
            exit(1)
    else:
        learning_rate_vec = [0.01]
        print('learning_rate: [0.01]')
    if 'gamma' in params:
        gamma_vec = params['gamma']
        if isinstance(gamma_vec, float):
            gamma_vec = [gamma_vec]
        elif isinstance(gamma_vec, list):
            if not isinstance(gamma_vec[0], float):
                print('gamma_vec is not a list of float')
                exit(1)
        else:
            print('gamma_vec is not a float or a list of float')
            exit(1)
    else:
        gamma_vec = [0.9]
        print('gamma: [0.9]')
    if 'space_dimension' in params:
        space_dimension_vec = params['space_dimension']
        if isinstance(space_dimension_vec, int):
            space_dimension_vec = [space_dimension_vec]
        elif isinstance(space_dimension_vec, list):
            if not isinstance(space_dimension_vec[0], int):
                print('space_dimension_vec is not a list of int')
                exit(1)
        else:
            print('space_dimension_vec is not a int or a list of int')
            exit(1)
    else:
        space_dimension_vec = [6]
        print('space_dimension_vec: [6]')
    if 'history_len' in params:
        history_len_vec = params['history_len']
        if isinstance(history_len_vec, int):
            history_len_vec = [history_len_vec]
        elif isinstance(history_len_vec, list):
            if not isinstance(history_len_vec[0], int):
                print('history_len_vec is not a list of int')
                exit(1)
        else:
            print('history_len_vec is not a int or a list of int')
            exit(1)
    else:
        history_len_vec = [10]
        print('history_len_vec: [10]')
    if 'single_threshold' in params:
        single_threshold_vec = params['single_threshold']
        if isinstance(single_threshold_vec, float):
            single_threshold_vec = [single_threshold_vec]
        elif isinstance(single_threshold_vec, list):
            if not isinstance(single_threshold_vec[0], float):
                print('single_threshold_vec is not a list of float')
                exit(1)
        else:
            print('single_threshold_vec is not a float or a list of float')
            exit(1)
    else:
        single_threshold_vec = [0.01]
        print('gamma: [0.01]')
    if 'use_reward' in params:
        use_reward_vec = params['use_reward']
        if isinstance(use_reward_vec, bool):
            use_reward_vec = [use_reward_vec]
        elif isinstance(use_reward_vec, list):
            if not isinstance(use_reward_vec[0], bool):
                print('use_reward_vec is not a list of bool')
                exit(1)
        else:
            print('use_reward_vec is not a bool or a list of bool')
            exit(1)
    else:
        single_threshold_vec = [False]
        print('gamma: [False]')

    data = datetime.datetime.now()
    test_name = name_space + 'tests'
    
    log_repo_path = path + '/log' 
    models_repo_path = path + '/model/'

    test_number = 0
    total_test = len(max_episode_steps_vec) * len(learning_rate_vec) * len(gamma_vec) * len(space_dimension_vec) * len(history_len_vec) * len(single_threshold_vec) * len(noise_sigma_vec) * len(use_reward_vec)
    print('Total tests: ' + str(total_test))
    for max_episode_steps in max_episode_steps_vec:
        for space_dimension in space_dimension_vec:
            for history_len in history_len_vec:
                for single_threshold in single_threshold_vec:
                    for learning_rate in learning_rate_vec:
                        for gamma in gamma_vec:
                            for noise_sigma in noise_sigma_vec:
                                for use_reward in use_reward_vec:
                                    if 'name_space' in params:
                                        name_space = params['name_space'] + '/'
                                    else:
                                        name_space = ''
                                    saving_name = name_space + '/' + str(max_episode_steps) + '/' + str(space_dimension) + '/' + str(history_len) + '/' + str(single_threshold) + '/' + str(learning_rate) + '/' + str(gamma) + '/' + str(noise_sigma) + '/' + str(use_reward)
                                    model_path = models_repo_path + '/' + saving_name
                                    log_path   = log_repo_path + '/' + saving_name

                                    # env = gym.make('GeneralEnv-v0', 
                                    #                 epoch_len = max_episode_steps,
                                    #                 max_episode_steps=max_episode_steps,
                                    #                 space_dimension=space_dimension,
                                    #                 history_len=history_len,
                                    #                 single_threshold=single_threshold,
                                    #                 use_reward=use_reward,
                                    #                 )
                                    env = gym.make('RealHistoryFakeEnv-v0', 
                                                    action_type='increment_value', 
                                                    epoch_len = max_episode_steps,
                                                    history_len=history_len,
                                                    use_reward=use_reward,
                                                    max_episode_steps=max_episode_steps)

                                    n_actions = env.action_space.shape[-1]
                                    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_sigma * np.ones(n_actions))
                                    model = TD3("MlpPolicy", 
                                                env, 
                                                verbose=verbose,
                                                action_noise=action_noise,
                                                learning_rate=learning_rate,
                                                tensorboard_log=log_path,
                                                gamma=gamma,
                                                )
                                    if 'model_save_freq' in params:
                                        checkpoint_callback = CheckpointCallback(
                                            save_freq=params['model_save_freq'],
                                            save_path=model_path + '/',
                                            name_prefix=name_space,
                                        )
                                        model.learn(total_timesteps=params['total_timesteps'], 
                                                    log_interval=1, 
                                                    callback=checkpoint_callback,
                                                    )
                                        model.save(model_path)

                                    else:
                                        model.learn(total_timesteps=params['total_timesteps'], 
                                                log_interval=1, 
                                                )
                                        model.save(model_path)
                                
