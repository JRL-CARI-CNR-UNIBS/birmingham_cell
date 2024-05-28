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

    max_epoch_steps = 500

    env = gym.make('TOFConnectionEnv-v0', 
                    epoch_len = max_epoch_steps,
                    max_episode_steps=max_epoch_steps)

    obs, info = env.reset()
    action = [0]   

    steps_vec = []

    for i in range(10):
        steps = 0
        success = False
        while (not success) and (steps < max_epoch_steps):
            obs, reward, success, truncated, info = env.step(action)
            steps += 1  
        steps_vec.append(steps)
    
    print('Final steps vector: ' + str(steps_vec))

    exit(0)
