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

separated = True
rospack = rospkg.RosPack()
pack_path = rospack.get_path('birmingham_cell_tests')

info_type = 'insert'
env = gym.make('ForceGraspEnv-v0',
               info_type=info_type)

obs, info = env.reset()

if info_type == 'grasp':
    print('Info type grasp')
    model = TD3.load(pack_path + '/model/grasp_model.zip')
elif info_type == 'insert':
    print('Info type insert')
    model = TD3.load(pack_path + '/model/insert_model.zip')
else:
    print('Info type error')

steps = 0
success = False

max_epoch_steps = 500
end_steps = []
for i in range(100):
    steps = 0
    success = False
    while (not success) and (steps < max_epoch_steps):
    # while (steps < max_epoch_steps):
        action, _states = model.predict(obs)
        obs, reward, success, truncated, info = env.step(action)
        steps += 1  
    end_steps.append(steps)
    env.reset()

print(end_steps)
