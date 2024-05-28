#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import birmingham_envs
import sys

from stable_baselines3 import TD3, SAC, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList

import rospkg
import os
import yaml
import datetime

class SuccessCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SuccessCallback, self).__init__(verbose)

    def _on_step(self) -> bool:        
        if 'episode' in self.locals['infos'][0]:
            episode_rewards = self.locals['infos'][0]['episode']['r']
            if episode_rewards >= 1:
                self.success_reached = True
                print(f"Success reached with reward: {episode_rewards}")
                return False  # Questo interrompe l'allenamento
        return True  # Continua l'allenamento

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
    print('name_space: ' + str(name_space))

    if 'verbose' in params:
        verbose = params['verbose']
    else:
        verbose = 0          
    print('verbose: ' + str(verbose))

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
    print('noise_sigma: ' + str(noise_sigma_vec))

    if 'train_freq' in params:
        if isinstance(params['train_freq'], dict):
            if 'step' in params['train_freq']:
                if isinstance(params['train_freq']['step'], int):
                    train_freq_vec = [(params['train_freq']['step'], 'step')]
                elif isinstance(params['train_freq']['step'], list):
                    if not isinstance(params['train_freq']['step'][0], int):
                        print('train_freq_vec is not a list of int')
                        exit(1)
                    train_freq_vec = []
                    for value in params['train_freq']['step']:
                        train_freq_vec.append((value,'step'))
                else:
                    print('train_freq_vec is not a int or a list of int')
                    exit(1)
            elif 'episode':
                if isinstance(params['train_freq']['episode'], int):
                    train_freq_vec = [(params['train_freq']['episode'], 'episode')]
                elif isinstance(params['train_freq']['episode'], list):
                    if not isinstance(params['train_freq']['episode'][0], int):
                        print('train_freq_vec is not a list of int')
                        exit(1)
                    train_freq_vec = []
                    for value in params['train_freq']['episode']:
                        train_freq_vec.append((value,'episode'))
                else:
                    print('train_freq_vec is not a int or a list of int')
                    exit(1)
        else:
            print('train_freq param is not a dict')
            exit(1)
    else:
        train_freq_vec = [1]
    print('train_freq_vec: ' + str(train_freq_vec))

    if 'max_epoch_steps' in params:
        max_epoch_steps_vec = params['max_epoch_steps']
        if isinstance(max_epoch_steps_vec, int):
            env_tymax_epoch_steps_vecpe_vec = [max_epoch_steps_vec]
        elif isinstance(max_epoch_steps_vec, list):
            if not isinstance(max_epoch_steps_vec[0], int):
                print('max_epoch_steps_vec is not a list of int')
                exit(1)
        else:
            print('max_epoch_steps_vec is not a int or a list of int')
            exit(1)
    else:
        max_epoch_steps_vec = [25]
    print('max_epoch_steps_vec: ' + str(max_epoch_steps_vec))
    
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
    print('learning_rate_vec: ' + str(learning_rate_vec))

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
    print('gamma_vec: ' + str(gamma_vec))

    if 'env_type' in params:
        env_type_vec = params['env_type']
        if isinstance(env_type_vec, str):
            env_type_vec = [env_type_vec]
        elif isinstance(env_type_vec, list):
            if not isinstance(env_type_vec[0], str):
                print('env_type_vec is not a list of string')
                exit(1)
        else:
            print('env_type_vec is not a string or a list of string')
            exit(1)
    else:
        print('No env_type')
        exit(1)
    print('env_type_vec: ' + str(env_type_vec))

    data = datetime.datetime.now()
    test_name = name_space + 'tests'
    
    log_repo_path = path + '/log' 
    models_repo_path = path + '/model/'

    test_number = 0
    total_test = len(env_type_vec) * len(max_epoch_steps_vec) * len(learning_rate_vec) * len(gamma_vec) * len(noise_sigma_vec)
    print('Total tests: ' + str(total_test))
    for env_type in env_type_vec:
        for max_epoch_steps in max_epoch_steps_vec:
            for learning_rate in learning_rate_vec:
                for gamma in gamma_vec:
                    for noise_sigma in noise_sigma_vec:
                        for train_freq in train_freq_vec:
                            test_number += 1
                            print('Test ' + str(test_number))
                            # model_name = test_name + '_' + str(max_epoch_steps) + '_' + str(learning_rate) + '_' + str(gamma)
                            # log_name = test_name + '_' + str(max_epoch_steps) + '_' + str(learning_rate) + '_' + str(gamma)
                            if 'name_space' in params:
                                name_space = params['name_space'] + '/'
                            else:
                                name_space = ''
                            save_name = name_space + env_type + '/' + str(max_epoch_steps) + '/' + str(learning_rate) + '/' + str(gamma) + '/' + str(noise_sigma) + '/' + str(train_freq)
                            model_path = models_repo_path + '/' + save_name
                            log_path = log_repo_path + '/' + save_name
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
                                                epoch_len = max_epoch_steps,
                                                max_episode_steps=max_epoch_steps)
                            elif env_type == 'static_connection':
                                env = gym.make('StaticConnectionEnv-v0', 
                                                action_type='target_value', 
                                                epoch_len = max_epoch_steps,
                                                max_episode_steps=max_epoch_steps)
                            elif env_type == 'realistic_fake':
                                print('In realistic_fake')
                                env = gym.make('RealisticFakeEnv-v0', 
                                                action_type='increment_value', 
                                                epoch_len = max_epoch_steps,
                                                max_episode_steps=max_epoch_steps)
                            elif env_type == 'static_real_fake':
                                print('In static_real_fake')
                                env = gym.make('StaticRealFakeEnv-v0', 
                                                action_type='target_value', 
                                                epoch_len = max_epoch_steps,
                                                max_episode_steps=max_epoch_steps)
                            elif env_type == 'realistic_history_fake':
                                print('In realistic_history_fake')
                                env = gym.make('RealHistoryFakeEnv-v0', 
                                                action_type='increment_value', 
                                                epoch_len = max_epoch_steps,
                                                max_episode_steps=max_epoch_steps)
                            elif env_type == 'action_param_history_fake':
                                print('In action_param_history_fake')
                                env = gym.make('RealHistoryFakeEnv-v0', 
                                                action_type='target_value', 
                                                epoch_len = max_epoch_steps,
                                                max_episode_steps=max_epoch_steps)
                            elif env_type == 'realistic_force_fake':
                                print('In realistic_force_fake')
                                env = gym.make('RealisticForceFakeEnv-v0', 
                                                action_type='increment_value', 
                                                epoch_len = max_epoch_steps,
                                                max_episode_steps=max_epoch_steps)
                            elif env_type == 'generic_real_force_fake':
                                print('In generic_real_force_fake')
                                env = gym.make('GenericRealForceFakeEnv-v0', 
                                                action_type='increment_value', 
                                                epoch_len = max_epoch_steps,
                                                max_episode_steps=max_epoch_steps)
                            elif env_type == 'realistic_fake_pos':
                                print('In ' + env_type)
                                env = gym.make('RealisticFakeEnv2-v0', 
                                                action_type='increment_value', 
                                                epoch_len = max_epoch_steps,
                                                obs_type = 'pos',
                                                max_episode_steps=max_epoch_steps)
                            elif env_type == 'realistic_fake_param_pos':
                                print('In ' + env_type)
                                env = gym.make('RealisticFakeEnv2-v0', 
                                                action_type='increment_value', 
                                                epoch_len = max_epoch_steps,
                                                obs_type = 'param_pos',
                                                max_episode_steps=max_epoch_steps)
                            elif env_type == 'realistic_fake_pos_param':
                                print('In ' + env_type)
                                env = gym.make('RealisticFakeEnv2-v0', 
                                                action_type='increment_value', 
                                                epoch_len = max_epoch_steps,
                                                obs_type = 'pos_param',
                                                max_episode_steps=max_epoch_steps)
                            elif env_type == 'realistic_fake_param':
                                print('In ' + env_type)
                                env = gym.make('RealisticFakeEnv2-v0', 
                                                action_type='increment_value', 
                                                epoch_len = max_epoch_steps,
                                                obs_type = 'param',
                                                max_episode_steps=max_epoch_steps)
                            elif env_type == 'realistic_fake_param_reward':
                                print('In ' + env_type)
                                env = gym.make('RealisticFakeEnv2-v0', 
                                                action_type='increment_value', 
                                                epoch_len = max_epoch_steps,
                                                obs_type = 'param_reward',
                                                max_episode_steps=max_epoch_steps)
                            elif env_type == 'realistic_fake_param_pos_reward':
                                print('In ' + env_type)
                                env = gym.make('RealisticFakeEnv2-v0', 
                                                action_type='increment_value', 
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
                                                epoch_len = max_epoch_steps,
                                                max_episode_steps=max_epoch_steps)
                            elif env_type == 'generic_real_fake':
                                print('In generic_real_fake')
                                env = gym.make('GenericRealFakeEnv-v0', 
                                                action_type='increment_value', 
                                                epoch_len = max_epoch_steps,
                                                # learning_starts = 100000,
                                                max_episode_steps=max_epoch_steps)
                            elif env_type == 'real_pos_feedback':
                                print('In real_pos_feedback')
                                env = gym.make('RealPosFeedbackFakeEnv-v0', 
                                                epoch_len = max_epoch_steps,
                                                max_episode_steps=max_epoch_steps)
                            elif env_type == 'general':
                                print('In general')
                                env = gym.make('GeneralEnv-v0', 
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
                                        train_freq = train_freq,
                                        learning_starts=1,
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
                                    success_callback = SuccessCallback()
                                    model.learn(total_timesteps=params['total_timesteps'], 
                                            log_interval=1, 
                                            callback=success_callback,
                                            )
                            
