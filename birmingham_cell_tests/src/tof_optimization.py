#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import birmingham_envs
import rospkg

rospack = rospkg.RosPack()
path = rospack.get_path('birmingham_cell_tests')

distance_threshold = 0.02
force_threshold = 100
debug_mode = False
step_print = False
only_pos_success = True
save_data = True
tree_name = 'tof_can_peg_in_hole'
test_name = 'tof_test'
data_repo_path = path + '/data/' + test_name + '_logs'

env = gym.make('FakeEnv-v0',
            action_type='increment_value')

env = gym.make('ConnectionEnv-v0', 
                action_type='increment_value', 
                distance_threshold=distance_threshold,
                force_threshold=force_threshold,
                debug_mode=debug_mode,
                step_print=step_print,
                only_pos_success=only_pos_success,
                data_file_name=test_name,
                tree_name=tree_name)
    
env.reset()

success = False
step = 0
max_step = 500
action = np.zeros(env.action_space.shape)

while ((not success) and (step < max_step)):
    obs, rew, success, trunc, info = env.step(action)
    step += 1

                    
