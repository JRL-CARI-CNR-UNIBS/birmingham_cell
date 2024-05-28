#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import birmingham_envs
import sys

from stable_baselines3 import TD3, SAC, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback,BaseCallback, CallbackList

import rospkg
import os
import yaml
import copy


class SuccessCallback(BaseCallback):
    def __init__(self, max_step, verbose=0):
        super(SuccessCallback, self).__init__(verbose)
        self.num_steps = 0
        self.max_step = max_step

    def _on_step(self) -> bool:        
        self.num_steps += 1
        if 'episode' in self.locals['infos'][0]:
            if self.num_steps >= self.max_step:
                print('Not success.')
                print('*********************************************************************************************************************************************************************')
            else:
                print('Success reached with ' + str(self.num_steps) + ' steps')
                print('*********************************************************************************************************************************************************************')
            return False  # Questo interrompe l'allenamento
        return True  # Continua l'allenamento

if __name__ == '__main__':

    max_epoch_steps = 500

    # env = gym.make('StaticRealFakeEnv-v0', 
    #                 action_type='target_value', 
    #                 epoch_len = max_epoch_steps,
    #                 max_episode_steps=max_epoch_steps)
    env = gym.make('ConnectionEnv-v0', 
                    action_type='target_value', 
                    epoch_len = max_epoch_steps,
                    max_episode_steps = max_epoch_steps)

    verbose = 0
    learning_rate = 0.01
    noise_sigma = 0.1
    gamma = 0.9
    train_freq = (1, 'step')
    total_timesteps = 500
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_sigma * np.ones(n_actions))
    
    for i in range(10):
        env.reset()

        model = TD3("MlpPolicy", 
            env, 
            verbose=verbose,
            action_noise=action_noise,
            learning_rate=learning_rate,
            # tensorboard_log=log_path,
            gamma=gamma,
            train_freq = train_freq,
            learning_starts=0,
            )
        
        success_callback = SuccessCallback(max_step=max_epoch_steps)
        model.learn(total_timesteps=total_timesteps, 
                log_interval=1, 
                callback=success_callback,
                )    
        

    exit(0)
