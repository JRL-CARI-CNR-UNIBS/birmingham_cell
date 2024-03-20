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


class EasyEnv(gym.Env):
    
    def __init__(
        self,
        node_name: str = 'Environment',
        package_name: str = 'birmingham_cell_tests',
        action_type: str = 'target_value',
        data_file_name: str = 'td3_tests',
        env_dimension: int = 1
    ) -> None:
        rospy.init_node(node_name)

        rospack = rospkg.RosPack()
        self.package_path = rospack.get_path(package_name)
        self.data_file_name = data_file_name + '.ods'
        self.step_number = 0
        self.start_obj_pos = None
        self.start_tar_pos = None
        self.last_action = None
        self.param_history = []
        self.action_type = action_type

        self.env_dimension = env_dimension

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

        print(self.all_param_names)
        print(self.param_lower_bound)
        print(self.param_upper_bound)
        print(self.initial_param_values)
        print(self.action_space)
        print(self.max_variations)
   
    def _get_obs(self) -> Dict[str, np.array]:
        observation = np.concatenate([np.array(self.param_values),np.subtract(self.correct_grasp_pos.tolist(),self.param_values)])
        self.obs = observation
        return observation

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[Dict[str, np.array], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        
        self.param_values = self.initial_param_values
        self.correct_grasp_pos = self._sample_object()
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
        object_position = np.full((self.env_dimension,), 0.0)
        noise = self.np_random.uniform(self.param_lower_bound, self.param_upper_bound)
        object_position += noise
        return object_position

    def _is_success(self) -> np.array:
        if (self._distance(self.correct_grasp_pos,self.param_values) < 0.001):
            success = True
        else:
            success = False
        return np.array(success, dtype=bool)

    def step(self, action: np.array) -> Tuple[Dict[str, np.array], float, bool, bool, Dict[str, Any]]:
        self.last_action = action
        self.step_number += 1
        if self.action_type == 'target_value':
            self.param_values = action.tolist()
        if self.action_type == 'increment_value':
            self.param_values = np.add(self.param_values, self.max_variations * action)
            self.param_values = np.clip(self.param_values, self.param_lower_bound, self.param_upper_bound)

        observation = self._get_obs()

        reward = self._get_reward()
        terminated = bool(self._is_success())
        truncated = False
        info = {"is_success": terminated}

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.array]:
        return 

    def _distance(self, pos1: np.array, pos2: np.array) -> float:
        return np.linalg.norm(np.array(pos1)-np.array(pos2))
    
    def _get_reward(self) -> float:
        reward = 1
        reward -= self._distance(
            self.correct_grasp_pos.tolist(),
            self.param_values
        ) * 10

#        print('obs ' + str(self.obs))
#        print('values ' + str(self.param_values))
#        print('action ' + str(self.last_action))
#        print('reward ' + str(reward))

        return reward