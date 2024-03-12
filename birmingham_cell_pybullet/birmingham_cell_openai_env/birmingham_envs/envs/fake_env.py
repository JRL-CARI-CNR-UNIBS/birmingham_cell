#!/usr/bin/env python3
from inspect import TPFLAGS_IS_ABSTRACT
from typing import Any, Dict, Optional, Tuple

import rospkg
import rospy
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
# from gymnasium.utils import seeding
import pyexcel_ods3 as od

import tf

from skills_util_msgs.srv import RunTree
from pybullet_simulation.srv import SpawnModel
from pybullet_simulation.srv import DeleteModel
from pybullet_simulation.srv import SaveState
from pybullet_simulation.srv import RestoreState
from pybullet_simulation.srv import DeleteState
from geometry_msgs.msg import Pose


class FakeEnv(gym.Env):
    
    def __init__(
        self,
        node_name: str = 'Connection_env',
        package_name: str = 'birmingham_cell_tests',
        trees_path: str = '/config/trees',
        tree_name: str = 'can_peg_in_hole',
        object_name: str = 'can',
        target_name: str = 'hole',
        distance_threshold: float = 0.02,
        force_threshold: float = 50,
        torque_threshold: float = 100,
        action_type: str = 'target_value',
        randomized_tf: list = ['can_grasp', 'hole_insertion'],
        debug_mode: bool = False,
        start_epoch_number: int = 0,
        data_file_name: str = 'td3_tests'
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
        self.epoch_number = start_epoch_number
        self.data_file_name = data_file_name + '.ods'
        self.step_number = 0
        self.start_obj_pos = None
        self.start_tar_pos = None
        self.last_action = None
        
        # arguments to define
        self.param_lower_bound = []
        self.param_upper_bound = []
        self.param_names_to_value_index = {}
        self.param_to_avoid_index = {}
        self.param_values = [0,0,0,0,0,0]
        self.initial_param_values = [0,0,0,0,0,0]
        self.obj_pos = []
        self.obj_rot = []
        self.tar_pos = []
        self.tar_rot = []
        self.obj_to_grasp_pos = []
        self.obj_to_grasp_rot = []
        self.tar_to_insertion_pos = []
        self.tar_to_insertion_rot = []
        self.initial_distance = None
        self.final_distance = None
        self.all_param_names = []
        self.param_history = []
        self.max_variations = []
              # Definisco la zona in cui possono essere posizionati gli oggetti di scena
        self.work_space_range_low  = np.array([0.3, -0.4, 0])
        self.work_space_range_high = np.array([0.6,  0.4, 0])
        self.obj_to_grasp_pos_default = np.array([0, 0, 0.06])
        self.tar_to_insert_pos_default = np.array([0, 0, 0.07])
        self.relative_grasp_correction = np.array([0, 0, 0])
        self.relative_inser_correction = np.array([0, 0, 0])
        self.relative_correct_grasp_pos = [0,0,0.07]
        self.relative_correct_insert_pos = [0,0,0.07]

        observation, _ = self.reset()  # required for init; seed can be changed later
        rospy.loginfo("Reset done")
        observation_shape = observation.shape
        self.observation_space = spaces.Box(-1, 1, shape=observation_shape, dtype=np.float64)
        
        self.all_param_names = ['x1','y1','z1','x2','y2','z2']
        self.param_lower_bound = [-0.05,-0.05,-0.05,-0.05,-0.05,-0.05]
        self.param_upper_bound = [ 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

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
        
        # salvo i dati ottenuti dal'epoca precedente 
        if self.param_history:
            self.epoch_number += 1
            data = []
            data.append(self.all_param_names + ['reward','iteration'])
            for index in range(len(self.param_history)):
                data.append(self.param_history[index])
            param_history_ods = od.get_data(self.package_path + "/data/" + self.data_file_name)
            param_history_ods.update({str(self.epoch_number): data})
            od.save_data(self.package_path + "/data/" + self.data_file_name, param_history_ods)
            self.param_history.clear()
            self.step_number = 0
        else:
            rospy.logwarn('Nothing to save')

        self.start_tar_pos = self._sample_target()
        self.start_obj_pos = self._sample_object()

        low_limit = [-0.02, -0.02, 0.0]
        high_limit = [0.02, 0.02, 0.02]

        initial_error_grasp_pos = np.ndarray.tolist(self.np_random.uniform(low_limit, high_limit))
        initial_error_insert_pos = np.ndarray.tolist(self.np_random.uniform(low_limit, high_limit))
        self.initial_grasp_pos = np.ndarray.tolist(np.add(self.relative_correct_grasp_pos,initial_error_grasp_pos))
        self.initial_insert_pos = np.ndarray.tolist(np.add(self.relative_correct_insert_pos,initial_error_insert_pos))

        self.current_grasp_pos = self.initial_grasp_pos
        self.current_insert_pos = self.initial_insert_pos
        self.param_values = [0,0,0,0,0,0]
        observation = self._get_obs()
        info = {"is_success": False}
        return observation, info
    
    def _sample_target(self) -> np.array:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, 0.0])
        noise = self.np_random.uniform(self.work_space_range_low, self.work_space_range_high)
        goal += noise
        return goal

    def _sample_object(self) -> np.array:
        """Randomize start position of object."""
        finish = False
        while not finish:
            object_position = np.array([0.0, 0.0, 0.0])
            noise = self.np_random.uniform(self.work_space_range_low, self.work_space_range_high)
            object_position += noise
            if (self._distance(self.start_tar_pos,object_position) > 0.25):
                finish = True
        return object_position

    def _is_success(self) -> np.array:
        if (self._distance(self.current_grasp_pos,self.relative_correct_grasp_pos) < 0.01 and
            self._distance(self.current_insert_pos,self.relative_correct_insert_pos) < 0.01):
            success = True
        else:
            success = False
        return np.array(success, dtype=bool)

    def step(self, action: np.array) -> Tuple[Dict[str, np.array], float, bool, bool, Dict[str, Any]]:
        self.last_action = action
        self.step_number += 1
        print('  ' + str(self.step_number))
        if self.debug_mode: self._print_action(action)

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
            self.current_grasp_pos = np.add(self.initial_grasp_pos, self.param_values[0:3])
            self.current_insert_pos = np.add(self.initial_insert_pos, self.param_values[3:6])

        observation = self._get_obs()

        reward = self._get_reward()
        terminated = bool(self._is_success())
        truncated = False
        info = {"is_success": terminated}

        # save data
        if self.param_history:
            data = []
            data.append(self.all_param_names + ['reward','iteration'])
            for index in range(len(self.param_history)):
                data.append(self.param_history[index])
            param_history_ods = od.get_data(self.package_path + "/data/" + self.data_file_name)
            param_history_ods.update({str(self.epoch_number): data})
            od.save_data(self.package_path + "/data/" + self.data_file_name, param_history_ods)
        else:
            rospy.logwarn('Nothing to save')



        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.array]:
        return 

    def _distance(self, pos1: np.array, pos2: np.array) -> float:
        return np.linalg.norm(np.array(pos1)-np.array(pos2))
    
    def _get_reward(self) -> float:
        # if (self._distance(self.current_grasp_pos, self.relative_correct_grasp_pos)<0.01):
        #     dist1 = self._distance(self.current_grasp_pos, self.relative_correct_grasp_pos)
        #     dist2 = self._distance(self.current_insert_pos, self.relative_correct_insert_pos)
        #     reward = 1
        #     reward -= dist1 * 25
        #     reward -= dist2 * 25
        #     print('In dist < 0.1. Dist1: ' + str(dist1) + ', Dist2: ' + str(dist2))
        #     print('Reward: ' + str(reward))
        # else:
        #     reward = 0.5
        #     dist_grasp = self._distance(self.current_grasp_pos[0:2], self.relative_correct_grasp_pos[0:2])
        #     dist_equal_grasp_insert = self._distance(self.current_grasp_pos[0:2], self.current_insert_pos[0:2])
        #     print('In dist > 0.1. dist_grasp: ' + str(dist_grasp) + 'dist_equal_grasp_insert: ' + str(dist_equal_grasp_insert))
        #     reward = 0.5
        #     reward -= dist_grasp
        #     reward -= dist_equal_grasp_insert
        #     print('Reward: ' + str(reward))

        reward = 1
        reward -= self._distance(self.current_grasp_pos,self.relative_correct_grasp_pos)
        reward -= self._distance(self.current_insert_pos,self.relative_correct_insert_pos)

        # Qua riempio lo storico dei parametri e il relativo reward
        rew_step_n = np.array([reward,self.step_number]).tolist()
        param_values = self.param_values.tolist()
        last_action = self.last_action.tolist()
        self.param_history.append(param_values + rew_step_n + last_action)

        print('current_grasp_pos ' + str(self.current_grasp_pos))
        print('current_insert_pos ' + str(self.current_insert_pos))
        print('correct_grasp_pos ' + str(self.relative_correct_grasp_pos))
        print('correct_insert_pos ' + str(self.relative_correct_insert_pos))
        print('grasp distance ' + str(self._distance(self.current_grasp_pos,self.relative_correct_grasp_pos)))
        print('insert distance ' + str(self._distance(self.current_insert_pos,self.relative_correct_insert_pos)))
        print('reward')

        return reward

    def _print_action(self, action) -> None:
        print(' ')
        print('ACTION____________________________________________________________________')
        for param_name in self.param_names_to_value_index.keys():
            if len(self.param_names_to_value_index[param_name]) == 1:
                print(param_name + ': ' + str(action[self.param_names_to_value_index[param_name][0]]))
            else:
                param_len = len(self.param_names_to_value_index[param_name])
                start_index = self.param_names_to_value_index[param_name][0]
                print(param_name + ': ' + str(action[start_index:start_index+param_len]))
        print('--------------------------------------------------------------------------')
        print(' ')

    def _print_obs(self, observation) -> None:
        print(' ')
        print('OBSERVATION_______________________________________________________________')
        for param_name in self.param_names_to_value_index.keys():
            if len(self.param_names_to_value_index[param_name]) == 1:
                print(param_name + ': ' + str(observation[self.param_names_to_value_index[param_name][0]]))
            else:
                param_len = len(self.param_names_to_value_index[param_name])
                start_index = self.param_names_to_value_index[param_name][0]
                print(param_name + ': ' + str(observation[start_index:start_index+param_len]))

        start_index = len(self.param_values)
        print('obj_to_grasp_pos: ' + str(observation[start_index:start_index+3]))
        print('tar_to_insertion_pos: ' + str(observation[start_index+3:start_index+6]))
        # print('max_wrench: ' + str(observation[start_index+6:start_index+12]))
        print('--------------------------------------------------------------------------')
        print(' ')
