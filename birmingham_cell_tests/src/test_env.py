#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import birmingham_envs
import sys

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

import rospkg
import os
import yaml


params = {}

if 'distance_threshold' in params:
    distance_threshold = params['distance_threshold']
else:
    distance_threshold = 0.02


if 'force_threshold' in params:
    force_threshold = params['force_threshold']
else:
    force_threshold = 100


if 'debug_mode' in params:
    debug_mode = params['debug_mode']
else:
    debug_mode = False


if 'step_print' in params:
    step_print = params['step_print']
else:
    step_print = False


if 'only_pos_success' in params:
    only_pos_success = params['only_pos_success']
else:
    only_pos_success = True

max_epoch_steps = 1000

env = gym.make('ConnectionEnv-v0', 
                distance_threshold=distance_threshold,
                force_threshold=force_threshold,
                debug_mode=debug_mode,
                step_print=step_print,
                only_pos_success=only_pos_success,
                max_episode_steps=max_epoch_steps)
