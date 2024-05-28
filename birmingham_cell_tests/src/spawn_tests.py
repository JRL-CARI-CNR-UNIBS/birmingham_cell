#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import birmingham_envs

from stable_baselines3 import TD3
# from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

import rospkg
import os

# env = gym.make('GeneralEnv-v0', 
#                 epoch_len = 25,
#                 max_episode_steps=10000,
#                 space_dimension=6,
#                 history_len=10,
#                 single_threshold=0.01,
#                 use_reward=False,
#                 )


env = gym.make('StaticConnectionEnv-v0',
               action_type='target_value', 
               max_episode_steps=25, 
               obj_model_name='can',
               tar_model_name='hole',
            #    obj_model_height=0.1,
            #    obj_model_length=0.04,
            #    obj_model_width =0.04,
            #    tar_model_height=0.06,
            #    tar_model_length=0.045,
            #    tar_model_width =0.045,
               debug_mode=False)

for i in range(10):
    print(i+1)
    env.reset()

exit(0)
action = [0.0,0.0,0.0,0.0,0.0,0.0]

os.system('roslaunch birmingham_cell_tests load_test_params.launch')

env.step(np.array(action))