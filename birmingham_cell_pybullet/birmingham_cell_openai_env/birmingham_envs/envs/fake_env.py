#!/usr/bin/env python3
from inspect import TPFLAGS_IS_ABSTRACT
from typing import Any, Dict, Optional, Tuple

import rospkg
import rospy
# import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
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
        self.last_action = None
        
        # arguments to define
        self.param_lower_bound = []
        self.param_upper_bound = []
        self.param_names_to_value_index = {}
        self.param_to_avoid_index = {}
        self.param_values = [0,0,0,0,0,0]
        self.initial_param_values = [0,0,0,0,0,0]
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
        self.correct_grasp_pos = [0,0,0.07]
        self.correct_insert_pos = [0,0,0.07]

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
#        print('param_values ' + str(self.param_values))
#        print('current_grasp_pos ' + str(self.current_grasp_pos))
#        print('current_insert_pos ' + str(self.current_insert_pos))
        observation = np.concatenate([np.array(self.param_values),np.array(self.current_grasp_pos),np.array(self.current_insert_pos)])
        return observation

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[Dict[str, np.array], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        
        low_limit = [-0.02, -0.02, 0.0]
        high_limit = [0.02, 0.02, 0.02]

        initial_error_grasp_pos = np.ndarray.tolist(self.np_random.uniform(low_limit, high_limit))
        initial_error_insert_pos = np.ndarray.tolist(self.np_random.uniform(low_limit, high_limit))
        self.initial_grasp_pos = np.ndarray.tolist(np.add(self.correct_grasp_pos,initial_error_grasp_pos))
        self.initial_insert_pos = np.ndarray.tolist(np.add(self.correct_insert_pos,initial_error_insert_pos))

        self.current_grasp_pos = self.initial_grasp_pos
        self.current_insert_pos = self.initial_insert_pos
        self.param_values = [0,0,0,0,0,0]
        observation = self._get_obs()
        info = {"is_success": False}
        return observation, info
    
    def _is_success(self) -> np.array:
        if (self._distance(self.current_grasp_pos,self.correct_grasp_pos) < 0.01 and
            self._distance(self.current_insert_pos,self.correct_insert_pos) < 0.01):
            success = True
        else:
            success = False
        return np.array(success, dtype=bool)

    def step(self, action: np.array) -> Tuple[Dict[str, np.array], float, bool, bool, Dict[str, Any]]:
        self.last_action = action
        self.step_number += 1

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

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.array]:
        return 

    def _distance(self, pos1: np.array, pos2: np.array) -> float:
        return np.linalg.norm(np.array(pos1)-np.array(pos2))
    
    def _get_reward(self) -> float:
        grasp_zone = str()

        # se il gripper è abbastanza vicino alla pos corretta lo considero in presa
        if (self._distance(self.current_grasp_pos, self.correct_grasp_pos)<0.01): 
            grasp_zone = 'successfully_grasp'
        # se il gripper è sotto la posizione corretta per più di un cm e non si allontana dal 
        # centro per più di 2 cm lo considero in collisione
        elif(self.current_grasp_pos-self.correct_grasp_pos < -0.01 and # 
            self._distance(self.current_grasp_pos[0:2],self.correct_grasp_pos[0:2]) < 0.02):
            grasp_zone = 'collision'
        # in ogni altro caso non sono in presa ne in collisione
        else:
            grasp_zone = 'free'

        if (grasp_zone == 'successfully_grasp'):
            # se sono in presa la bontà aumenta con l'allineamento gripper oggetto
            dist1 = self._distance(self.current_grasp_pos, self.correct_grasp_pos)
            dist2 = self._distance(self.current_insert_pos, self.correct_insert_pos)
            reward = 1
            reward -= dist1 * 15
            reward -= dist2 * 15
        elif(grasp_zone == 'collision'):
            # se sono in collisione la bontà aumenta se sono allineato ma anche se mi sposto verso l'alto
            # per l'insert si deve allineare
            dist_xy_grasp = self._distance(self.current_grasp_pos[0:2], self.correct_grasp_pos[0:2])
            insertion_movement = np.linalg.norm(np.multiply(self.last_action[3:6],self.max_variations[3:6]))
            dist_flor_to_grasp = self.current_grasp_pos[2]
            reward = 0.5
            reward += dist_flor_to_grasp/3
            reward -= dist_xy_grasp * 4.4
            reward -= insertion_movement * 4.4
        elif(grasp_zone == 'free'):
            # se sono libero la bontà aumenta se mi avvicino all'oggetto
            insertion_movement = np.linalg.norm(np.multiply(self.last_action[3:6],self.max_variations[3:6]))
            dist_grasp = np.linalg.norm(self.current_grasp_pos)
            reward = 0.3
            reward -= insertion_movement
            reward -= dist_grasp

        # reward = 1
        # reward -= self._distance(self.current_grasp_pos,self.correct_grasp_pos) * 4
        # reward -= self._distance(self.current_insert_pos,self.correct_insert_pos) * 4

        # print('action ' + str(self.last_action))
        # print('modification ' + str(np.multiply(self.last_action, self.max_variations)))
        # print('param ' + str(self.param_values))
        # print('current_grasp_pos ' + str(self.current_grasp_pos))
        # print('current_insert_pos ' + str(self.current_insert_pos))
        # print('correct_grasp_pos ' + str(self.correct_grasp_pos))
        # print('correct_insert_pos ' + str(self.correct_insert_pos))
        # print('grasp distance ' + str(self._distance(self.current_grasp_pos,self.correct_grasp_pos)))
        # print('insert distance ' + str(self._distance(self.current_insert_pos,self.correct_insert_pos)))
        # print('reward ' + str(reward))

        return reward
