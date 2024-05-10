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
    if 'distance_threshold' in params:
        distance_threshold = params['distance_threshold']
    else:
        distance_threshold = 0.02
        print('distance_threshold: 0.02')
    if 'force_threshold' in params:
        force_threshold = params['force_threshold']
    else:
        force_threshold = 100
        print('force_threshold: 100')
    if 'debug_mode' in params:
        debug_mode = params['debug_mode']
    else:
        debug_mode = False
        print('debug_mode: False')
    if 'step_print' in params:
        step_print = params['step_print']
    else:
        step_print = False
        print('step_print: False')
    if 'only_pos_success' in params:
        only_pos_success = params['only_pos_success']
    else:
        only_pos_success = True
        print('only_pos_success: True')
    if 'noise_sigma' in params:
        noise_sigma_vec = params['noise_sigma']
    else:
        noise_sigma_vec = [0.1]
        print('noise_sigma: [0.1]')

    if 'max_epoch_steps' in params:
        max_epoch_steps_vec = params['max_epoch_steps']
    else:
        max_epoch_steps_vec = [25]
        print('max_epoch_steps: [25]')
    if 'learning_rate' in params:
        learning_rate_vec = params['learning_rate']
    else:
        learning_rate_vec = [0.01]
        print('learning_rate: [0.01]')
    if 'gamma' in params:
        gamma_vec = params['gamma']
    else:
        gamma_vec = [0.9]
        print('gamma: [0.9]')
    if 'env_type' in params:
        env_type_vec = params['env_type']
    else:
        print('No env_type')
        exit(1)
   

    data = datetime.datetime.now()
    test_name = name_space + 'tests'
    
    log_repo_path = path + '/log' 
    models_repo_path = path + '/model/'

    test_number = 0
    total_test = len(params['max_epoch_steps']) * len(params['learning_rate']) * len(params['gamma'])
    print('Total tests: ' + str(total_test))
    for env_type in params['env_type']:
        for max_epoch_steps in max_epoch_steps_vec:
            for learning_rate in learning_rate_vec:
                for gamma in gamma_vec:
                    for noise_sigma in noise_sigma_vec:
                        test_number += 1
                        print('Test ' + str(test_number))
                        # model_name = test_name + '_' + str(max_epoch_steps) + '_' + str(learning_rate) + '_' + str(gamma)
                        # log_name = test_name + '_' + str(max_epoch_steps) + '_' + str(learning_rate) + '_' + str(gamma)
                        if 'name_space' in params:
                            name_space = params['name_space'] + '/'
                        else:
                            name_space = ''
                        model_name = name_space + env_type + '/'  + str(max_epoch_steps) + '/' + str(learning_rate) + '/' + str(gamma) + '/' + str(noise_sigma)
                        log_name = name_space + env_type + '/' + str(max_epoch_steps) + '/' + str(learning_rate) + '/' + str(gamma) + '/' + str(noise_sigma)
                        model_path = models_repo_path + '/' + model_name
                        log_path = log_repo_path + '/' + log_name
                        if env_type == 'fake':
                            env = gym.make('FakeEnv-v0',
                                        action_type='increment_value',
                                        epoch_len = max_epoch_steps,
                                        max_episode_steps=max_epoch_steps)
                        elif env_type == 'easy':
                            env = gym.make('EasyEnv-v0',
                                        action_type='increment_value', 
                                        env_dimension=1,
                                        obs_type='history',
                                        history_len=10,
                                        max_episode_steps=max_epoch_steps)
                        elif env_type == 'connection':
                            env = gym.make('ConnectionEnv-v0', 
                                            action_type='increment_value', 
                                            distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'realistic_fake':
                            print('In realistic_fake')
                            env = gym.make('RealisticFakeEnv-v0', 
                                            action_type='increment_value', 
                                            # distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'realistic_history_fake':
                            print('In realistic_history_fake')
                            env = gym.make('RealHistoryFakeEnv-v0', 
                                            action_type='increment_value', 
                                            epoch_len = max_epoch_steps,
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'realistic_force_fake':
                            print('In realistic_force_fake')
                            env = gym.make('RealisticForceFakeEnv-v0', 
                                            action_type='increment_value', 
                                            # distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'generic_real_force_fake':
                            print('In generic_real_force_fake')
                            env = gym.make('GenericRealForceFakeEnv-v0', 
                                            action_type='increment_value', 
                                            # distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'realistic_fake_pos':
                            print('In ' + env_type)
                            env = gym.make('RealisticFakeEnv2-v0', 
                                            action_type='increment_value', 
                                            # distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            obs_type = 'pos',
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'realistic_fake_param_pos':
                            print('In ' + env_type)
                            env = gym.make('RealisticFakeEnv2-v0', 
                                            action_type='increment_value', 
                                            # distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            obs_type = 'param_pos',
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'realistic_fake_pos_param':
                            print('In ' + env_type)
                            env = gym.make('RealisticFakeEnv2-v0', 
                                            action_type='increment_value', 
                                            # distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            obs_type = 'pos_param',
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'realistic_fake_param':
                            print('In ' + env_type)
                            env = gym.make('RealisticFakeEnv2-v0', 
                                            action_type='increment_value', 
                                            # distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            obs_type = 'param',
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'realistic_fake_param_reward':
                            print('In ' + env_type)
                            env = gym.make('RealisticFakeEnv2-v0', 
                                            action_type='increment_value', 
                                            # distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            obs_type = 'param_reward',
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'realistic_fake_param_pos_reward':
                            print('In ' + env_type)
                            env = gym.make('RealisticFakeEnv2-v0', 
                                            action_type='increment_value', 
                                            # distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            obs_type = 'param_pos_reward',
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'realistic_fake_param_pos_reward_history':
                            print('In ' + env_type)
                            if 'history_len' in params:
                                history_len_vec = params['history_len']
                            else:
                                history_len_vec = [10]
                            for history_len in history_len_vec:
                                env = gym.make('RealisticFakeEnv2-v0', 
                                                action_type='increment_value', 
                                                # distance_threshold=distance_threshold,
                                                force_threshold=force_threshold,
                                                debug_mode=debug_mode,
                                                step_print=step_print,
                                                only_pos_success=only_pos_success,
                                                epoch_len = max_epoch_steps,
                                                obs_type = 'param_pos_reward_history',
                                                max_episode_steps=max_epoch_steps,
                                                history_len=history_len)
                                n_actions = env.action_space.shape[-1]
                                action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                                log_path = log_path + '/' + str(history_len)
                                model = TD3("MlpPolicy", 
                                            env, 
                                            verbose=verbose,
                                            action_noise=action_noise,
                                            learning_rate=learning_rate,
                                            tensorboard_log=log_path,
                                            gamma=gamma,
                                            ) 
                                model.learn(total_timesteps=params['total_timesteps'], 
                                            log_interval=1, 
                                            )
                            continue
                        elif env_type == 'realistic_fake_pos_reward_history':
                            print('In ' + env_type)
                            if 'history_len' in params:
                                history_len_vec = params['history_len']
                            else:
                                history_len_vec = [10]
                            for history_len in history_len_vec:
                                env = gym.make('RealisticFakeEnv2-v0', 
                                                action_type='increment_value', 
                                                # distance_threshold=distance_threshold,
                                                force_threshold=force_threshold,
                                                debug_mode=debug_mode,
                                                step_print=step_print,
                                                only_pos_success=only_pos_success,
                                                epoch_len = max_epoch_steps,
                                                obs_type = 'pos_reward_history',
                                                max_episode_steps=max_epoch_steps,
                                                history_len=history_len)
                                n_actions = env.action_space.shape[-1]
                                action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                                log_path = log_path + '/' + str(history_len)
                                model = TD3("MlpPolicy", 
                                            env, 
                                            verbose=verbose,
                                            action_noise=action_noise,
                                            learning_rate=learning_rate,
                                            tensorboard_log=log_path,
                                            gamma=gamma,
                                            ) 
                                model.learn(total_timesteps=params['total_timesteps'], 
                                            log_interval=1, 
                                            )
                            continue
                        elif env_type == 'realistic_fake_param_reward_history':
                            print('In ' + env_type)
                            if 'history_len' in params:
                                history_len_vec = params['history_len']
                            else:
                                history_len_vec = [10]
                            for history_len in history_len_vec:
                                env = gym.make('RealisticFakeEnv2-v0', 
                                                action_type='increment_value', 
                                                # distance_threshold=distance_threshold,
                                                force_threshold=force_threshold,
                                                debug_mode=debug_mode,
                                                step_print=step_print,
                                                only_pos_success=only_pos_success,
                                                epoch_len = max_epoch_steps,
                                                obs_type = 'param_reward_history',
                                                max_episode_steps=max_epoch_steps,
                                                history_len=history_len)
                                n_actions = env.action_space.shape[-1]
                                action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                                log_path = log_path + '/' + str(history_len)
                                model = TD3("MlpPolicy", 
                                            env, 
                                            verbose=verbose,
                                            action_noise=action_noise,
                                            learning_rate=learning_rate,
                                            tensorboard_log=log_path,
                                            gamma=gamma,
                                            ) 
                                model.learn(total_timesteps=params['total_timesteps'], 
                                            log_interval=1, 
                                            )
                            continue
                        elif env_type == 'realistic_fake_param_pos_and_reward_history':
                            print('In ' + env_type)
                            if 'history_len' in params:
                                history_len_vec = params['history_len']
                            else:
                                history_len_vec = [10]
                            for history_len in history_len_vec:
                                env = gym.make('RealisticFakeEnv2-v0', 
                                                action_type='increment_value', 
                                                # distance_threshold=distance_threshold,
                                                force_threshold=force_threshold,
                                                debug_mode=debug_mode,
                                                step_print=step_print,
                                                only_pos_success=only_pos_success,
                                                epoch_len = max_epoch_steps,
                                                obs_type = 'param_pos_and_reward_history',
                                                max_episode_steps=max_epoch_steps,
                                                history_len=history_len)
                                n_actions = env.action_space.shape[-1]
                                action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                                log_path = log_path + '/' + str(history_len)
                                model = TD3("MlpPolicy", 
                                            env, 
                                            verbose=verbose,
                                            action_noise=action_noise,
                                            learning_rate=learning_rate,
                                            tensorboard_log=log_path,
                                            gamma=gamma,
                                            ) 
                                model.learn(total_timesteps=params['total_timesteps'], 
                                            log_interval=1, 
                                            )
                            continue
                        elif env_type == 'random_real_fake':
                            print('In random_real_fake')
                            env = gym.make('RandomRealFakeEnv-v0', 
                                            action_type='increment_value', 
                                            # distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'generic_real_fake':
                            print('In generic_real_fake')
                            env = gym.make('GenericRealFakeEnv-v0', 
                                            action_type='increment_value', 
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            max_episode_steps=max_epoch_steps)
                        else:
                            print('Env_type not in the possible env list.')
                            exit(0)  

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
                           
