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


class EasyEnv(gym.Env):
    
    def __init__(
        self,
        node_name: str = 'Connection_env',
        package_name: str = 'birmingham_cell_tests',
        action_type: str = 'target_value',
        data_file_name: str = 'td3_tests'
    ) -> None:
        # rospy.init_node(node_name)

        # rospack = rospkg.RosPack()
        self.package_path ='/home/gauss/projects/personal_ws/src/birmingham_cell/birmingham_cell_tests'
        # self.package_path = rospack.get_path(package_name)
        self.data_file_name = data_file_name + '.ods'
        self.step_number = 0
        self.start_obj_pos = None
        self.start_tar_pos = None
        self.last_action = None
        self.param_history = []
        self.action_type = action_type

        self.env_dimension = 6

        self.all_param_names = np.full((self.env_dimension,), 'x')
        # self.all_param_names = ['x1','y1','z1','x2','y2','z2']
        # self.all_param_names = ['x']

        self.param_lower_bound = np.full((self.env_dimension,), -0.05)
        self.param_upper_bound = np.full((self.env_dimension,),  0.05)
        # self.param_lower_bound = [-0.05,-0.05,-0.05,-0.05,-0.05,-0.05]
        # self.param_upper_bound = [ 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        # self.param_lower_bound = [-0.05]
        # self.param_upper_bound = [ 0.05]

        self.initial_param_values = np.full((self.env_dimension,), 0.0)
        # self.initial_param_values = [0,0,0,0,0,0]
        # self.initial_param_values = [0]
        self.epoch_number = 0


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
            # self.action_space = spaces.Box(low=np.array(self.param_lower_bound), high=np.array(self.param_upper_bound), dtype=np.float64)
        else:
            rospy.logerr('The action type ' + action_type + ' is not supported.')
   
    def _get_obs(self) -> Dict[str, np.array]:
        # observation = np.concatenate([np.array(self.param_values),self.correct_grasp_pos,self.correct_insert_pos])
        observation = np.concatenate([np.array(self.param_values),np.subtract(self.correct_grasp_pos.tolist(),self.param_values)])
        self.obs = observation
        # observation = np.concatenate([np.array(self.param_values)])
        return observation

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[Dict[str, np.array], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        
        # salvo i dati ottenuti dal'epoca precedente 
        if self.param_history:
            self.epoch_number += 1
            data = []
            data.append(self.all_param_names.tolist() + ['reward','iteration'])
            for index in range(len(self.param_history)):
                data.append(self.param_history[index])
            param_history_ods = od.get_data(self.package_path + "/data/" + self.data_file_name)
            param_history_ods.update({str(self.epoch_number): data})
            od.save_data(self.package_path + "/data/" + self.data_file_name, param_history_ods)
            self.param_history.clear()
            self.step_number = 0
        else:
            rospy.logwarn('Nothing to save')

        # self.param_values = [0,0,0,0,0,0]
        self.param_values = self.initial_param_values
        self.correct_grasp_pos = self._sample_object()
        # self.correct_insert_pos = self._sample_target()
        observation = self._get_obs()
        info = {"is_success": False}
        return observation, info
    
    def _sample_target(self) -> np.array:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, 0.0])
        noise = self.np_random.uniform(self.param_lower_bound[3:6], self.param_upper_bound[3:6])
        goal += noise
        return goal

    def _sample_object(self) -> np.array:
        """Randomize start position of object."""
        # object_position = np.array([0.0, 0.0, 0.0])
        # noise = self.np_random.uniform(self.param_lower_bound[0:3], self.param_upper_bound[0:3])
        object_position = np.full((self.env_dimension,), 0.0)
        noise = self.np_random.uniform(self.param_lower_bound, self.param_upper_bound)
        # print(object_position)
        # print(noise)
        object_position += noise
        return object_position

    def _is_success(self) -> np.array:
        # if (self._distance(self.correct_grasp_pos,self.param_values[0:3]) < 0.001 and
        #     self._distance(self.correct_insert_pos,self.param_values[3:6]) < 0.001):
        if (self._distance(self.correct_grasp_pos,self.param_values) < 0.001):
            success = True
        else:
            success = False
        return np.array(success, dtype=bool)

    def step(self, action: np.array) -> Tuple[Dict[str, np.array], float, bool, bool, Dict[str, Any]]:
        self.last_action = action
        self.step_number += 1
        # print(str(self.epoch_number) + '  ' + str(self.step_number))
        # Settaggio dei nuovi parametri attraverso la action.
        # Se l'azione è il nuovo set di parametri
        if self.action_type == 'target_value':
            self.param_values = action.tolist()
        # Se l'azione è la variazione
        if self.action_type == 'increment_value':
            self.param_values = np.add(self.param_values, self.max_variations * action)
            self.param_values = np.clip(self.param_values, self.param_lower_bound, self.param_upper_bound)

        observation = self._get_obs()

        reward = self._get_reward()
        terminated = bool(self._is_success())
        # if self.last_action == 1.0 or self.last_action == -1.0:
        #     truncated = True
        # else:
        truncated = False
        info = {"is_success": terminated}

        # save data
        # if self.param_history:
        #     data = []
        #     data.append(self.all_param_names + ['reward','iteration'])
        #     for index in range(len(self.param_history)):
        #         data.append(self.param_history[index])
        #     param_history_ods = od.get_data(self.package_path + "/data/" + self.data_file_name)
        #     param_history_ods.update({str(self.epoch_number): data})
        #     od.save_data(self.package_path + "/data/" + self.data_file_name, param_history_ods)
        # else:
        #     rospy.logwarn('Nothing to save')

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.array]:
        return 

    def _distance(self, pos1: np.array, pos2: np.array) -> float:
        return np.linalg.norm(np.array(pos1)-np.array(pos2))
    
    def _get_reward(self) -> float:
        reward = 1
        reward -= self._distance(
            # self.correct_grasp_pos.tolist() + self.correct_insert_pos.tolist(),
            self.correct_grasp_pos.tolist(),
            self.param_values
        ) * 5
        # reward -= self._distance(self.correct_grasp_pos,self.param_values[0:3]) * 4
        # reward -= self._distance(self.correct_insert_pos,self.param_values[3:6]) * 4
        # print(reward)
        # Qua riempio lo storico dei parametri e il relativo reward
        rew_step_n = np.array([reward,self.step_number]).tolist()
        param_values = self.param_values.tolist()
        last_action = self.last_action.tolist()
        self.param_history.append(param_values + rew_step_n + last_action)
        return reward