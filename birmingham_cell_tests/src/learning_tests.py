#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import birmingham_envs
import sys

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.monitor import Monitor

import rospkg
import os
import yaml
import datetime


if __name__ == '__main__':

    params_path = sys.argv[1]

    
    rospack = rospkg.RosPack()
    pack_path = rospack.get_path('birmingham_cell_tests')
    
    file_path = pack_path + '/' + params_path

    with open(file_path) as file:
        params = yaml.safe_load(file)

    if 'total_timesteps' in params:
        total_timesteps = params['total_timesteps']
    else:
        total_timesteps = 1000000
    print('total_timesteps ' + str(total_timesteps))
    if 'max_epoch_steps' in params:
        max_epoch_steps = params['max_epoch_steps']
    else:
        max_epoch_steps = 25
    print('max_epoch_steps ' + str(max_epoch_steps))
    if 'learning_rate' in params:
        learning_rate = params['learning_rate']
    else:
        learning_rate = 0.01
    print('learning_rate ' + str(learning_rate))
    if 'gamma' in params:
        gamma = params['gamma']
    else:
        gamma = 0.9
    print('gamma ' + str(gamma))
    if 'history_len' in params:
        history_len = params['history_len']
    else:
        history_len = 10
    print('history_len ' + str(history_len))
    if 'verbose' in params:
        verbose = params['verbose']
    else:
        verbose = 0
    print('verbose ' + str(verbose))
    if 'model_save_freq' in params:
        model_save_freq = params['model_save_freq']
    else:
        model_save_freq = total_timesteps
    print('model_save_freq ' + str(model_save_freq))
    if 'name_space' in params:
        name_space = params['name_space']
    else:
        name_space = 'no_name_space'
    print('name_space ' + str(name_space))
    if 'noise_sigma' in params:
        noise_sigma = params['noise_sigma']
    else:
        noise_sigma = '0.1'
    print('noise_sigma ' + str(noise_sigma))

    log_path = pack_path + '/log' + '/' + name_space  + '/tt_' + str(total_timesteps) + '/mes_' + str(max_epoch_steps) + '/lr_' + str(learning_rate) + '/g' + str(gamma) + '/hl' + str(history_len)
    model_path = pack_path + '/model' + '/' + name_space  + '/tt_' + str(total_timesteps) + '/mes_' + str(max_epoch_steps) + '/lr_' + str(learning_rate) + '/g' + str(gamma) + '/hl' + str(history_len)

    env = gym.make('RealisticFakeEnv2-v0', 
                    action_type='increment_value', 
                    epoch_len = max_epoch_steps,
                    obs_type = 'param_reward',
                    max_episode_steps=max_epoch_steps,)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=history_len)
    # env = VecNormalize(env)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_sigma * np.ones(n_actions))

    checkpoint_callback = CheckpointCallback(
        save_freq=model_save_freq,
        save_path=model_path + '/',
    )

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
                callback=checkpoint_callback,
                )

    model.save(model_path)
