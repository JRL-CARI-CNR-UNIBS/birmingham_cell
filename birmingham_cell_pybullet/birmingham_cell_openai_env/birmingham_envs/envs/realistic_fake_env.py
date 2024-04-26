#!/usr/bin/env python3
from inspect import TPFLAGS_IS_ABSTRACT
from typing import Any, Dict, Optional, Tuple

import rospkg
import rospy
# import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import copy
# from gymnasium.utils import seeding
# import pyexcel_ods3 as od

# import tf

# from skills_util_msgs.srv import RunTree
# from pybullet_simulation.srv import SpawnModel
# from pybullet_simulation.srv import DeleteModel
# from pybullet_simulation.srv import SaveState
# from pybullet_simulation.srv import RestoreState
# from pybullet_simulation.srv import DeleteState
# from geometry_msgs.msg import Pose


class RealisticFakeEnv(gym.Env):
    
    def __init__(
        self,
        node_name: str = 'Realistic_fake_env',
        package_name: str = 'birmingham_cell_tests',
        trees_path: str = '/config/trees',
        tree_name: str = 'can_peg_in_hole',
        object_name: str = 'can',
        target_name: str = 'hole',
        distance_threshold: float = 0.005,
        force_threshold: float = 50,
        torque_threshold: float = 100,
        action_type: str = 'target_value',
        randomized_tf: list = ['can_grasp', 'hole_insertion'],
        debug_mode: bool = False,
        data_file_name: str = 'td3_tests',
        save_data: bool = False,
        step_print: bool = False,
        only_pos_success: bool = True,
        epoch_len: int = None,
        obj_pos_error: list = [0.0,0.0,0.0],
    ) -> None:
        rospy.init_node(node_name)

        rospack = rospkg.RosPack()
        self.package_path = rospack.get_path(package_name)
        self.trees_path = [self.package_path + trees_path]
        self.tree_name = tree_name
        self.object_name = object_name
        self.target_name = target_name
        self.distance_threshold = distance_threshold
        self.force_threshold = force_threshold
        self.torque_threshold = torque_threshold
        self.action_type = action_type   # currently available, target_value  increment_value  
        self.randomized_tf = randomized_tf
        self.debug_mode = debug_mode
        self.data_file_name = data_file_name + '.ods'
        self.save_data = save_data
        self.step_print = step_print
        self.only_pos_success = only_pos_success
        self.step_number = 0
        self.epoch_len = epoch_len
        self.epoch_steps = 0
        self.obj_pos_error = obj_pos_error
        
        # arguments to define
        self.start_obj_pos = None
        self.start_tar_pos = None
        self.last_action = None
        self.param_lower_bound = []
        self.param_upper_bound = []
        self.param_names_to_value_index = {}
        self.param_to_avoid_index = {}
        self.param_values = []
        self.initial_param_values = {}
        # self.tar_pos = []
        # self.tar_rot = []
        self.obj_to_grasp_pos = []
        self.obj_to_grasp_rot = []
        self.tar_to_insertion_pos = []
        self.tar_to_insertion_rot = []
        self.initial_distance = None
        self.final_distance = None
        self.all_param_names = []
        self.param_history = []
        self.max_variations = []
        self.observation = None
        self.obj_pos = None
        self.obj_rot = None
        self.tar_pos = None
        self.tar_rot = None
        self.old_obj_pos = None
        self.old_obj_rot = None
        self.old_tar_pos = None
        self.old_tar_rot = None
        self.gripper_grasp_pos = None
        self.gripper_grasp_rot = None
        self.initial_obj_to_grasp_pos = None
        self.insertion_pos_param = '/exec_params/actions/can_peg_in_hole/skills/move_to_hole_insertion/relative_position'
        self.init_par_val = []
        self.current_grasp_pos = None
        self.current_insert_pos = None

        # lettura dei parametri delle skill da modificare
        try:
            exec_ns = rospy.get_param('/skills_executer/skills_parameters_name_space')
        except:
            rospy.logerr('/skills_executer/skills_parameters_name_space not found')
            exit(0)

        try:
            learn_ns = rospy.get_param('/skills_learning/learning_parameter_name_space')
        except:
            rospy.logerr('/skills_learning/learning_parameter_name_space not found')
            exit(0)

        try:
            action_params = rospy.get_param('/' + learn_ns + '/actions')
        except:
            rospy.logerr('/' + learn_ns + '/actions')
            exit(0)
        
        n_env_action = 0
        for action_name in action_params:
            for skill_name in action_params[action_name]['skills']:
                for param_name in action_params[action_name]['skills'][skill_name]:
                    param_value = action_params[action_name]['skills'][skill_name][param_name]
                    complete_name = '/' + exec_ns + '/actions/' + action_name + '/skills/' + skill_name + '/' + param_name
                    self.param_names_to_value_index.update({complete_name: []})
                    self.initial_param_values[complete_name] = rospy.get_param(complete_name)
                    if isinstance(param_value[0], int) or isinstance(param_value[0], float):
                        if (len(param_value) == 2):
                            self.init_par_val.append(self.initial_param_values[complete_name])
                            n_env_action +=1
                            self.param_lower_bound.append(param_value[0])
                            self.param_upper_bound.append(param_value[1])
                            self.param_names_to_value_index[complete_name].append(n_env_action-1)
                            self.all_param_names.append('/' + action_name + '/' + skill_name + '/' + param_name)
                        else:
                            rospy.logerr('There is a problem with the structure of /' + 
                                         learn_ns + '/' + action_name + 
                                         '/skills/' + skill_name + 
                                         '/' + param_name)
                    elif isinstance(param_value[0], list):
                        for par_val in self.initial_param_values[complete_name]:
                            self.init_par_val.append(par_val)
                        self.param_to_avoid_index[complete_name] = []
                        for i in range(len(param_value)):
                            if (len(param_value[i]) == 2):
                                if param_value[i][0] == param_value[i][1]:
                                    self.param_to_avoid_index[complete_name].append(i)
                                else:
                                    self.param_lower_bound.append(param_value[i][0])
                                    self.param_upper_bound.append(param_value[i][1])
                                    self.param_names_to_value_index[complete_name].append(n_env_action)
                                    n_env_action +=1
                                    self.all_param_names.append('/' + action_name + '/' + skill_name + '/' + param_name + str(i))
                            else:
                                rospy.logerr('There is a problem with the structure of /' + 
                                            learn_ns + '/' + action_name + 
                                            '/skills/' + skill_name + 
                                            '/' + param_name)
                    else:
                        rospy.logerr('There is a problem with the structure of /' + 
                                     learn_ns + '/' + action_name + 
                                     '/skills/' + skill_name + 
                                     '/' + param_name)          
            
        # Definisco la zona in cui possono essere posizionati gli oggetti di scena
        self.work_space_range_low  = np.array([0.3, -0.4, 0])
        self.work_space_range_high = np.array([0.6,  0.4, 0])

        self.correct_grasp_pos = [0,0,0.04]
        self.correct_insert_pos = [0,0,0.15]
        
        self.correct_grasp_pos  = (np.array(self.correct_grasp_pos)  + np.array(self.obj_pos_error)).tolist()
        self.correct_insert_pos = (np.array(self.correct_insert_pos) + np.array(self.obj_pos_error)).tolist()
       

        observation, _ = self.reset()  # required for init; seed can be changed later
        rospy.loginfo("Reset done")
        observation_shape = observation.shape
        self.observation_space = spaces.Box(-1, 1, shape=observation_shape, dtype=np.float64)

        if self.action_type == 'target_value':
            self.action_space = spaces.Box(np.array(self.param_lower_bound),np.array(self.param_upper_bound), dtype=np.float64)
        # Se scegliamo di avere l'azione come una variazione
        elif self.action_type == 'increment_value':
            space_division = 10.0
            self.max_variations = (np.array(self.param_upper_bound)-np.array(self.param_lower_bound))/space_division
            self.action_space = spaces.Box(-1, 1, shape=(len(self.max_variations),), dtype=np.float64)
        else:
            rospy.logerr('The action type ' + action_type + ' is not supported.')
 
    def _get_obs(self) -> Dict[str, np.array]:
        observation = np.concatenate([np.array(self.param_values),np.array(self.current_grasp_pos),np.array(self.current_insert_pos)])
        return observation

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[Dict[str, np.array], Dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.epoch_steps = 0
        low_limit = [-0.02, -0.02, 0.0]
        high_limit = [0.02, 0.02, 0.02]

        initial_error_grasp_pos = np.ndarray.tolist(self.np_random.uniform(low_limit, high_limit))
        initial_error_insert_pos = np.ndarray.tolist(self.np_random.uniform(low_limit, high_limit))
        self.initial_grasp_pos = np.ndarray.tolist(np.add(self.correct_grasp_pos,initial_error_grasp_pos))
        self.initial_insert_pos = np.ndarray.tolist(np.add(self.correct_insert_pos,initial_error_insert_pos))

        self.current_grasp_pos  = copy.copy(self.initial_grasp_pos)
        self.current_insert_pos = copy.copy(self.initial_insert_pos)
        self.param_values = copy.copy(self.init_par_val)
        # self.param_values = [0,0,0,0,0,0]
        observation = self._get_obs()
        info = {"is_success": False}
        return observation, info
    
    def _is_success(self) -> np.array:
        if (self._distance(self.current_grasp_pos,self.correct_grasp_pos) < (self.distance_threshold / 2) and
            self._distance(self.current_insert_pos,self.correct_insert_pos) < (self.distance_threshold / 2)):
            success = True
            # print('grasp ' + str(self.current_grasp_pos))
            # print('insert ' + str(self.current_insert_pos))
        else:
            success = False
        return np.array(success, dtype=bool)

    def step(self, action: np.array) -> Tuple[Dict[str, np.array], float, bool, bool, Dict[str, Any]]:
        self.last_action = action
        self.epoch_steps += 1

        # Settaggio dei nuovi parametri attraverso la action.
        # Se l'azione è il nuovo set di parametri
        if self.action_type == 'target_value':
            self.param_values = action.tolist()
            self.current_grasp_pos = np.add(self.initial_grasp_pos, action[0:3])
            self.current_insert_pos = np.add(self.initial_insert_pos, action[3:6])
        # Se l'azione è la variazione
        if self.action_type == 'increment_value':
            self.param_values = np.add(self.param_values, np.multiply(action, self.max_variations))
            self.param_values = np.clip(self.param_values, self.param_lower_bound, self.param_upper_bound)
            self.current_grasp_pos[0] = self.initial_grasp_pos[0] + self.param_values[0]
            self.current_grasp_pos[1] = self.initial_grasp_pos[1] - self.param_values[1]
            self.current_grasp_pos[2] = self.initial_grasp_pos[2] - self.param_values[2]
            # self.current_grasp_pos = np.add(self.initial_grasp_pos, self.param_values[0:3])
            self.current_insert_pos[0] = self.initial_insert_pos[0] + self.param_values[3]
            self.current_insert_pos[1] = self.initial_insert_pos[1] - self.param_values[4]
            self.current_insert_pos[2] = self.initial_insert_pos[2] - self.param_values[5]
            # self.current_insert_pos = np.add(self.initial_insert_pos, self.param_values[3:6])

        observation = self._get_obs()

        success = bool(self._is_success())

        if ((self.epoch_len is not None) and success):
            single_reward = self._get_reward()
            remain_step = self.epoch_len - self.epoch_steps
            # reward = remain_step * single_reward
            reward = single_reward + remain_step
            # print('Success!')
            # print('  Single reward: ' + str(single_reward))
            # print('  Remain step: ' + str(remain_step))
            # print('  Reward: ' + str(reward))
        else:
            reward = self._get_reward()

        terminated = success
        truncated = False
        info = {"is_success": terminated}

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.array]:
        return 

    def _distance(self, pos1: np.array, pos2: np.array) -> float:
        return np.linalg.norm(np.array(pos1)-np.array(pos2))
    
    def _get_reward(self) -> float:
        grasp_zone = str()

        # se il gripper è abbastanza vicino alla pos corretta lo considero in presa
        if ((self._distance(self.current_grasp_pos[2], self.correct_grasp_pos[2]) < 0.01) and
            (self._distance(self.current_grasp_pos[0:2],self.correct_grasp_pos[0:2]) < 0.015)): 
            grasp_zone = 'successfully_grasp'
        # se il gripper è sotto la posizione corretta per più di un cm e non si allontana dal 
        # centro per più di 2 cm lo considero in collisione
        elif((self.current_grasp_pos[2]-self.correct_grasp_pos[2] < -0.01) and 
            (self._distance(self.current_grasp_pos[0:2],self.correct_grasp_pos[0:2])) < 0.02):
            grasp_zone = 'collision'
        else:
            grasp_zone = 'free'
        # in ogni altro caso non sono in presa ne in collisione

        if (grasp_zone == 'successfully_grasp'):
            # se sono in presa la bontà aumenta con l'allineamento gripper oggetto
            dist1 = self._distance(self.current_grasp_pos, self.correct_grasp_pos)
            dist2 = self._distance(self.current_insert_pos, self.correct_insert_pos)
            reward = 1
            reward -= dist1 * 10# max 0.0180. x10 -> 0.18
            reward -= dist2 * 10 # max 0.05. x10 -> 0.5
            # 0.32 < reward < 1
        elif(grasp_zone == 'collision'):
            # se sono in collisione la bontà aumenta se sono allineato ma anche se mi sposto verso l'alto
            # per l'insert si deve allineare
            dist_xy_grasp = self._distance(self.current_grasp_pos[0:2], self.correct_grasp_pos[0:2])
            insertion_movement = np.linalg.norm(np.multiply(self.last_action[3:6],self.max_variations[3:6]))
            dist_flor_to_grasp = self.current_grasp_pos[2]
            reward = 0
            reward += dist_flor_to_grasp * 5 # max 0.06. x5 -> 0.3
            reward -= dist_xy_grasp * 7.5 # max 0.02. x7.5 -> 0.2
            reward -= insertion_movement * 5.5 # max 0.018. x5.5 -> ~0.1
            # -0.3 < reward < 0.3
        else:
            # se sono libero la bontà aumenta se mi avvicino all'oggetto
            insertion_movement = np.linalg.norm(np.multiply(self.last_action[3:6],self.max_variations[3:6]))
            dist_grasp = np.linalg.norm(self.current_grasp_pos)
            reward = -0.3
            reward -= insertion_movement * 6 # max 0.018. x7.9 -> 0.15   Not perfect
            reward -= dist_grasp * 6 # max 0.07. x7.9 -> 0.55  Not perfect
            # -1 < reward < -0.3

        # reward = 1
        # reward -= self._distance(self.current_grasp_pos, self.correct_grasp_pos) * 10
        # reward -= self._distance(self.current_insert_pos, self.correct_insert_pos) * 10

        return reward
