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



env = gym.make('ConnectionEnv-v0',
               action_type='increment_value', 
               max_episode_steps=25, 
               obj_model_name='cylinder',
               tar_model_name='cylinder_hole',
               obj_model_height=0.1,
               obj_model_length=0.04,
               obj_model_width =0.04,
               tar_model_height=0.06,
               tar_model_length=0.045,
               tar_model_width =0.045,
               debug_mode=False)

env.reset()

action = [0.0,0.0,0.0,0.0,0.0,0.0]

os.system('roslaunch birmingham_cell_tests load_test_params.launch')

env.step(np.array(action))