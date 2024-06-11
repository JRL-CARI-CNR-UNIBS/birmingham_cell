#!/usr/bin/env python3
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from inspect import TPFLAGS_IS_ABSTRACT
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import birmingham_envs

from stable_baselines3 import TD3
# from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

import rospkg
import copy

rospack = rospkg.RosPack()
pack_path = rospack.get_path('birmingham_cell_tests')

env = gym.make('ConnectionForcesEnv-v0')

obs, info = env.reset()

sampling_freq = 250
grasp_max_time = 1
insert_max_time = 2
grasp_recording_lenght = grasp_max_time * sampling_freq
insert_recording_lenght = insert_max_time * sampling_freq
grasp_data_lengh = grasp_recording_lenght * 6
insert_data_lengh = insert_recording_lenght * 6
print(grasp_data_lengh)
print(insert_data_lengh)

print(len(obs))
grasp_param = obs[:2]
grasp_pose = obs[4:6]
grasp_forces = obs[8:(8+grasp_data_lengh)]
grasp_obs = np.concatenate([obs[:2],obs[4:6],obs[8:(8+grasp_data_lengh)]])
insert_param = obs[2:4]
insert_pose = obs[6:8]
insert_pose = obs[(8+grasp_data_lengh):(8+grasp_data_lengh+insert_data_lengh)]
insert_obs = np.concatenate([obs[2:4],obs[6:8],obs[(8+grasp_data_lengh):(8+grasp_data_lengh+insert_data_lengh)]])

grasp_model = TD3.load(pack_path + '/model/grasp_model.zip')
insert_model = TD3.load(pack_path + '/model/insert_model.zip')

steps = 0
success = False

# action = np.array([0,0,0,0,0,0])
# env.step(action)
max_epoch_steps = 50
end_steps = []
for i in range(10):
    steps = 0
    success = False
    while (not success) and (steps < max_epoch_steps):
    # while (steps < max_epoch_steps):
        grasp_action, _states = grasp_model.predict(grasp_obs)
        insert_action, _states = insert_model.predict(insert_obs)
        action = np.concatenate([grasp_action,insert_action])
        obs, reward, success, truncated, info = env.step(action)
        grasp_obs = np.concatenate([obs[:2],obs[4:6],obs[8:(8+grasp_data_lengh)]])
        insert_obs = np.concatenate([obs[2:4],obs[6:8],obs[(8+grasp_data_lengh):(8+grasp_data_lengh+insert_data_lengh)]])
        print(action)
        steps += 1  
    print(steps)
    end_steps.append(steps)
    env.reset()

print(end_steps)

