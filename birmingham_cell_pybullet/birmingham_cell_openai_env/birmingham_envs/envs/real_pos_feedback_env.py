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


class RealPosFeedbackFakeEnv(gym.Env):
    
    def __init__(
        self,
        action_type: str = 'increment_value',
        epoch_len: int = None,
        random_error: bool = False,
    ) -> None:
        self.action_type = action_type   # currently available, target_value  increment_value  
        self.epoch_len = epoch_len
        self.epoch_steps = 0

        self.initial_param_values = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.param_lower_bound    = [-0.1,-0.1,-0.1,-0.1,-0.1,-0.1]
        self.param_upper_bound    = [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.max_variations       = [0.01,0.01,0.01,0.01,0.01,0.01]

        self.initial_grasp_pos_lower_bound  = [-0.05,-0.05,-0.05]
        self.initial_grasp_pos_upper_bound  = [ 0.05, 0.05, 0.05]
        self.initial_insert_pos_lower_bound = [-0.05,-0.05,-0.05]
        self.initial_insert_pos_upper_bound = [ 0.05, 0.05, 0.05]

        self.obj_pos_error_lower_bound = [-0.03,-0.03,-0.03]
        self.obj_pos_error_upper_bound = [ 0.03, 0.03, 0.03]
                
        # questo lo si potrà cambiare in funzione dell'oggetto che utiliziamo, partiamo dal tenerlo costante
        self.theoretical_correct_grasp_pos  = [0,0,0.04]
        self.theoretical_correct_insert_pos = [0,0,0.15]
        self.correct_grasp_area_lower_bound = [-0.005,-0.005,-0.005]
        self.correct_grasp_area_upper_bound = [ 0.005, 0.005, 0.005]
        self.grasp_area_lower_bound         = [ -0.02, -0.01, -0.02]
        self.grasp_area_upper_bound         = [  0.02,  0.01,  0.02]
        self.collision_area_lower_bound     = [ -0.06, -0.06, -0.12]
        self.collision_area_upper_bound     = [  0.06,  0.06, -0.02]
        self.insert_pos_z_lower_bound = 0.10
        
        self.random_error = random_error
        self.obj_pos_error = []
        self.correct_grasp_pos = self.theoretical_correct_grasp_pos
        self.correct_insert_pos = self.theoretical_correct_insert_pos

        self.grasp_zone = 0
        self.insert_zone = 0

        # self.correct_grasp_pos  = (np.array(self.correct_grasp_pos)  + np.array(self.obj_pos_error)).tolist()
        # self.correct_insert_pos = (np.array(self.correct_insert_pos) + np.array(self.obj_pos_error)).tolist()    

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
        observation = np.concatenate([np.array([self.grasp_zone]),np.array([self.insert_zone]),np.array(self.param_values),np.array(self.current_grasp_pos),np.array(self.current_insert_pos)])
        return observation

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[Dict[str, np.array], Dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.epoch_steps = 0

        if self.random_error:
            self.obj_pos_error = np.ndarray.tolist(self.np_random.uniform(self.obj_pos_error_lower_bound, self.obj_pos_error_upper_bound))
        else:
            self.obj_pos_error = [0.0,0.0,0.0]

        self.correct_grasp_pos = np.ndarray.tolist(np.subtract(np.array(self.theoretical_correct_grasp_pos),np.array(self.obj_pos_error)))

        initial_error_grasp_pos  = np.ndarray.tolist(self.np_random.uniform(self.initial_grasp_pos_lower_bound, self.initial_grasp_pos_upper_bound))
        initial_error_insert_pos = np.ndarray.tolist(self.np_random.uniform(self.initial_insert_pos_lower_bound, self.initial_insert_pos_upper_bound))
        self.initial_grasp_pos   = np.ndarray.tolist(np.add(self.theoretical_correct_grasp_pos,initial_error_grasp_pos))
        self.initial_insert_pos  = np.ndarray.tolist(np.add(self.theoretical_correct_insert_pos,initial_error_insert_pos))

        self.current_grasp_pos  = copy.copy(self.initial_grasp_pos)
        self.current_insert_pos = copy.copy(self.initial_insert_pos)
        self.param_values = copy.copy(self.initial_param_values)

        self.grasp_zone = self._get_grasp_zone()
        observation = self._get_obs()
        info = {"is_success": False}
        return observation, info
    
    def _is_success(self) -> np.array:
        if self.insert_zone == 3:
            success = True
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
            reward = single_reward + remain_step
        else:
            reward = self._get_reward()

        terminated = success
        truncated = False
        info = {"is_success": terminated}

        return observation, reward, terminated, truncated, info

    def _distance(self, pos1: np.array, pos2: np.array) -> float:
        return np.linalg.norm(np.array(pos1)-np.array(pos2))
    
    def _get_reward(self) -> float:
        self.grasp_zone = self._get_grasp_zone()
        self.insert_zone = self._get_insertion_zone()
        if self.grasp_zone == 0:
            # condizione in cui l'oggetto non è in presa ma neanche in collisione, cerco di avvicinarmi all'oggetto
            dist_grasp = np.linalg.norm(self.current_grasp_pos)
            reward = -0.5
            reward -= dist_grasp * 0.1
        elif self.grasp_zone == 1:
            # gripper e oggetto in collisione, devo spostarmi verso l'alto
            dist_xy_grasp = np.linalg.norm(self.current_grasp_pos[:2])
            dist_flor_to_grasp = self.current_grasp_pos[2]
            reward = -0.25
            reward -= dist_xy_grasp * 0.1
            reward += dist_flor_to_grasp * 0.1
        elif self.grasp_zone == 2: 
            # oggetto in presa ma in una posizione sbagliata, questo fa si che l'oggetto venga spostato durante l'operazione di afferraggio
            grasp_x_error = self.current_grasp_pos[0] - self.correct_grasp_pos[0]
            grasp_y_error = self.current_grasp_pos[1] - self.correct_grasp_pos[1]
            self.obj_x_grasp_movement = grasp_x_error
            self.obj_y_grasp_movement = -grasp_y_error
            reward = 0.25
            reward -= abs(grasp_x_error) * 0.1
            reward -= abs(grasp_y_error) * 0.1
        elif self.grasp_zone == 3:
            # oggetto in presa corretta, quindi non c'è movimento o è minimo. Focalizzarsi sulle forze di inserimento
            insert_x_error = self.current_insert_pos[0] - self.current_insert_pos[0]
            insert_y_error = self.current_insert_pos[1] - self.current_insert_pos[1]
            self.x_force = insert_x_error
            self.y_force = insert_y_error
            # print('insert' + str(self.insert_zone))
            if self.insert_zone == 1:
                reward = 0.5
                reward -= abs(self.x_force) *0.1
                reward -= abs(self.y_force) *0.1
                diff_between_poses = self._distance(np.subtract(self.current_grasp_pos[:2],self.correct_grasp_pos[:2]),np.subtract(self.current_insert_pos[:2],self.correct_insert_pos[:2]))
                reward -= diff_between_poses * 0.1
            elif self.insert_zone == 2:
                reward = 0.5
                reward += self.current_grasp_pos[2] * 0.1
            elif self.insert_zone == 3:
                reward = 1
                reward -= abs(self.x_force) *0.1
                reward -= abs(self.y_force) *0.1
        return reward

    def _get_grasp_zone(self) -> int:
        if (np.all(np.array(self.current_grasp_pos) >= np.add(np.array(self.correct_grasp_pos),np.array(self.correct_grasp_area_lower_bound))) & 
            np.all(np.array(self.current_grasp_pos) <= np.add(np.array(self.correct_grasp_pos),np.array(self.correct_grasp_area_upper_bound)))):
            # la presa è avvenuta nella zona corretta, quindi non vi sono movimenti dell'oggetto durante la presa
            grasp_zone = 3
        elif (np.all(np.array(self.current_grasp_pos) >= np.add(np.array(self.correct_grasp_pos),np.array(self.grasp_area_lower_bound))) & 
              np.all(np.array(self.current_grasp_pos) <= np.add(np.array(self.correct_grasp_pos),np.array(self.grasp_area_upper_bound)))):
            # vi è stata presa dell'oggetto ma uesto si è mosso, la presa va sistemata
            grasp_zone = 2
        elif (np.all(np.array(self.current_grasp_pos) >= np.add(np.array(self.correct_grasp_pos),np.array(self.collision_area_lower_bound))) & 
              np.all(np.array(self.current_grasp_pos) <= np.add(np.array(self.correct_grasp_pos),np.array(self.collision_area_upper_bound)))):
            # sono in collisione con l'oggetto
            grasp_zone = 1
        else:
            # non sono ne in presa ne in collisione
            grasp_zone = 0

        return grasp_zone

    def _get_insertion_zone(self) -> int:
        if self.grasp_zone < 3:
            # non posso avere un'inserzione che fornisca dati rilevanti perché il grasp è andato male 
            insert_zone = 0
        else:   
            if (self._distance(np.subtract(self.current_grasp_pos[:2],self.correct_grasp_pos[:2]),np.subtract(self.current_insert_pos[:2],self.correct_insert_pos[:2])) > 0.003):
                # grasp andato bene però posizione di grasp e di inserzione non sono relativamente simili quindi l'oggetto non dovrebbe essere incentrato
                # print('grasp_error' + str(np.subtract(self.current_grasp_pos[:2],self.correct_grasp_pos[:2])))
                # print('inser_error' + str(np.subtract(self.current_insert_pos[:2],self.correct_insert_pos[:2])))
                # print('distance   ' + str(self._distance(np.subtract(self.current_grasp_pos[:2],self.correct_grasp_pos[:2]),np.subtract(self.current_insert_pos[:2],self.correct_insert_pos[:2]))))
                # print('insert 1')
                insert_zone = 1
            elif (self.current_grasp_pos < self.correct_grasp_pos):
                # situazione in cui ho preso correttamente l'oggetto, l'ho anche centrato, ma questo non arriva sul fondo 
                insert_zone = 2
                # print('insert 2')
            else:
                # oggetto inserito completamente
                # print('insert 3')
                insert_zone = 3
        return insert_zone