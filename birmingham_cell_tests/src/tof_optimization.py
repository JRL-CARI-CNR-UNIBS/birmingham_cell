#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import birmingham_envs
import rospkg
import rospy

rospack = rospkg.RosPack()
path = rospack.get_path('birmingham_cell_tests')

obtimization_max_number = 10
distance_threshold = 0.02
force_threshold = 100
debug_mode = False
step_print = False
only_pos_success = True
save_data = True
tree_name = 'tof_can_peg_in_hole'
test_name = 'tof_test'
data_repo_path = path + '/data/' + test_name + '_data'

reward_param_name = '/exec_params/actions/can_peg_in_hole/total_reward'

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
obtimization_number = 0 

while (obtimization_number < obtimization_max_number):
    obs, rew, success, trunc, info = env.step(action)
    step += 1
    rospy.set_param(reward_param_name,float(rew))
    if (success or (step == max_step)):
        obtimization_number += 1
        print(step)
        step = 0 
        env.reset()
    

                    
