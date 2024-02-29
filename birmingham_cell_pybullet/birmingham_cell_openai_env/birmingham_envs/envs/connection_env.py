#!/usr/bin/env python3
from inspect import TPFLAGS_IS_ABSTRACT
from typing import Any, Dict, Optional, Tuple

import rospy
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
# from gymnasium.utils import seeding

import tf

from skills_util_msgs.srv import RunTree
from pybullet_simulation.srv import SpawnModel
from pybullet_simulation.srv import DeleteModel
from pybullet_simulation.srv import SaveState
from pybullet_simulation.srv import RestoreState
from pybullet_simulation.srv import DeleteState
from geometry_msgs.msg import Pose


class ConnectionEnv(gym.Env):
    
    def __init__(
        self,
        node_name: str = 'Connection_env',
        trees_path: list = ['/home/gauss/projects/personal_ws/src/birmingham_cell/birmingham_cell_tests/config/trees'],
        tree_name: str = 'can_peg_in_hole',
        object_name: str = 'can',
        target_name: str = 'hole',
        distance_threshold: float = 0.02,
        force_threshold: float = 50,
        torque_threshold: float = 100,
        action_type: str = 'target_value',
        randomized_tf: list = ['can_grasp', 'hole_insertion'],
        debug_mode: bool = False
    ) -> None:
        rospy.init_node(node_name)

        self.trees_path = trees_path
        self.tree_name = tree_name
        self.object_name = object_name
        self.target_name = target_name
        self.distance_threshold = distance_threshold
        self.force_threshold = force_threshold
        self.torque_threshold = torque_threshold
        self.action_type = action_type   # currently available, target_value  increment_value  
        self.randomized_tf = randomized_tf
        self.debug_mode = debug_mode

        # arguments to define
        self.param_lower_bound = []
        self.param_upper_bound = []
        self.param_names_to_value_index = {}
        self.param_to_avoid_index = {}
        self.param_values = []
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
                    if isinstance(param_value[0], int) or isinstance(param_value[0], float):
                        if (len(param_value) == 2):
                            n_env_action +=1
                            self.param_lower_bound.append(param_value[0])
                            self.param_upper_bound.append(param_value[1])
                            self.param_names_to_value_index[complete_name].append(n_env_action-1)
                        else:
                            rospy.logerr('There is a problem with the structure of /' + 
                                         learn_ns + '/' + action_name + 
                                         '/skills/' + skill_name + 
                                         '/' + param_name)
                    elif isinstance(param_value[0], list):
                        self.param_to_avoid_index[complete_name] = []
                        for i in range(len(param_value)):
                            if (len(param_value[i]) == 2):
                                if param_value[i][0] == param_value[i][1]:
                                    self.param_to_avoid_index[complete_name].append(i)
                                else:
                                    n_env_action +=1
                                    self.param_lower_bound.append(param_value[i][0])
                                    self.param_upper_bound.append(param_value[i][1])
                                    self.param_names_to_value_index[complete_name].append(n_env_action-1)
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

        # creo i client per i servizi pybullet
        rospy.loginfo("Wait for skills_util/run_tree service")
        rospy.wait_for_service('/skills_util/run_tree')
        self.run_tree_clnt = rospy.ServiceProxy('/skills_util/run_tree', RunTree)
        rospy.loginfo("Connected")
        rospy.loginfo("Wait for pybullet_spawn_model service")
        rospy.wait_for_service('/pybullet_spawn_model')
        self.spawn_model_clnt = rospy.ServiceProxy('/pybullet_spawn_model', SpawnModel)
        rospy.loginfo("Connected")
        rospy.loginfo("Wait for pybullet_delete_model service")
        rospy.wait_for_service('/pybullet_delete_model')
        self.delete_model_clnt = rospy.ServiceProxy('/pybullet_delete_model', DeleteModel)
        rospy.loginfo("Connected")
        rospy.loginfo("Wait for pybullet_save_state service")
        rospy.wait_for_service('/pybullet_save_state')
        self.save_state_clnt = rospy.ServiceProxy('/pybullet_save_state', SaveState)
        rospy.loginfo("Connected")
        rospy.loginfo("Wait for pybullet_restore_state service")
        rospy.wait_for_service('/pybullet_restore_state')
        self.restore_state_clnt = rospy.ServiceProxy('/pybullet_restore_state', RestoreState)
        rospy.loginfo("Connected")
        rospy.loginfo("Wait for pybullet_delete_state service")
        rospy.wait_for_service('/pybullet_delete_state')
        self.delete_state_clnt = rospy.ServiceProxy('/pybullet_delete_state', DeleteState)
        rospy.loginfo("Connected")

        # Setto il robot nella sua posizione iniziale e salvo lo stato pybullet
        self.run_tree_clnt.call('init_tree', self.trees_path)
        self.reset_state_name = 'reset'
        self.save_state_clnt.call(self.reset_state_name)
        self.step_state_name = 'step'
        rospy.loginfo("Init done")


        # Definisco la zona in cui possono essere posizionati gli oggetti di scena
        self.work_space_range_low  = np.array([0.3, -0.4, 0])
        self.work_space_range_high = np.array([0.6,  0.4, 0])

        # leggo come son state settate le tf iniziali
        self.initial_tf = rospy.get_param('/tf_params')


        self.tf_listener = tf.TransformListener()
        self.start_obj_pos = None
        self.start_tar_pos = None
        observation, _ = self.reset()  # required for init; seed can be changed later
        rospy.loginfo("Reset done")
        observation_shape = observation.shape
        self.observation_space = spaces.Box(float('-inf'), float('inf'), shape=observation_shape, dtype=np.float64)

        # Definizione dell'action space
        # Se scegliamo di avere l'azione uguale al valore che desideriamo 
        if self.action_type == 'target_value':
            self.action_space = spaces.Box(np.array(self.param_lower_bound),np.array(self.param_upper_bound), dtype=np.float64)
        # Se scegliamo di avere l'azione come una variazione
        elif self.action_type == 'increment_value':
            space_division = 10.0
            max_variations = (np.array(self.param_upper_bound)-np.array(self.param_lower_bound))/space_division
            self.action_space = spaces.Box(-max_variations,max_variations, dtype=np.float64)
        else:
            rospy.logerr('The action type ' + action_type + ' is not supported.')

    def _create_scene(self,obj_pos,tar_pos) -> None:
        """Create the scene."""
        object_names = []
        object_names.append(self.target_name)  
        object_names.append(self.object_name)
        self.delete_model_clnt.call(object_names)

        model_name = []
        pose = []
        fixed = []
        tar_pose = Pose()

        tar_pose.position.x = tar_pos[0]
        tar_pose.position.y = tar_pos[1]
        tar_pose.position.z = tar_pos[2]
        tar_pose.orientation.x = 0.0
        tar_pose.orientation.y = 0.0
        tar_pose.orientation.z = 0.0
        tar_pose.orientation.w = 1.0
        model_name.append(self.target_name)
        pose.append(tar_pose)
        fixed.append(True)

        obj_pose = Pose()
        obj_pose.position.x = obj_pos[0]
        obj_pose.position.y = obj_pos[1]
        obj_pose.position.z = obj_pos[2]
        obj_pose.orientation.x = 0.0
        obj_pose.orientation.y = 0.0
        obj_pose.orientation.z = 0.0
        obj_pose.orientation.w = 1.0
        model_name.append(self.object_name)
        pose.append(obj_pose)
        fixed.append(False)
        self.spawn_model_clnt.call(object_names, model_name, pose, fixed)

        # randomizzo le posizioni di grasp e di insertion
        new_tfs = []
        for old_tf in self.initial_tf:
            new_tf = copy.copy(old_tf)
            if new_tf['name'] in self.randomized_tf:
                low_limit = [-0.02, -0.02, -0.02]
                high_limit = [0.02, 0.02, 0.02]
                noise = np.ndarray.tolist(self.np_random.uniform(low_limit, high_limit))
                new_tf['position'] = np.ndarray.tolist(np.add(new_tf['position'], noise))
            new_tfs.append(new_tf)
        rospy.set_param('tf_params',new_tfs)

    def _get_obs(self) -> Dict[str, np.array]:
        # Come osservazione utilizzo le posizioni relative e il set di parametri
        self._update_info()
        observation = np.concatenate([np.array(self.param_values),np.array(self.obj_to_grasp_pos),np.array(self.tar_to_insertion_pos),np.array(self.max_wrench)])
        # observation = np.concatenate([np.array(self.param_values),np.array(self.obj_to_grasp_pos),np.array(self.tar_to_insertion_pos)])
        if self.debug_mode: self._print_obs(observation)
        return observation

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[Dict[str, np.array], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        
        # rimuovo gli oggetti della scena, riporto il robot nello stato iniziale e poi riaggiungo gli oggetti 
        # in una posizione casuale ma non sovrapposta
        object_names = [self.target_name, self.object_name]
        self.delete_model_clnt.call(object_names)
        self.delete_state_clnt.call([self.step_state_name])
        rospy.loginfo("Model and step_state deleted")

        self.restore_state_clnt.call(self.reset_state_name)
        rospy.loginfo("Reset_state restored")

        self.start_tar_pos = self._sample_target()
        self.start_obj_pos = self._sample_object()
        self._create_scene(self.start_obj_pos,self.start_tar_pos)
        rospy.loginfo("Scene created")

        self.save_state_clnt(self.step_state_name)

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
        self._update_info()
        if (self.final_distance < self.distance_threshold and
            self.max_wrench[0] < self.force_threshold and
            self.max_wrench[1] < self.force_threshold and
            self.max_wrench[3] < self.torque_threshold and
            self.max_wrench[4] < self.torque_threshold):
            success = True
        else:
            success = False
        return np.array(success, dtype=bool)

    def step(self, action: np.array) -> Tuple[Dict[str, np.array], float, bool, bool, Dict[str, Any]]:
        if self.debug_mode: self._print_action(action)
        # Settaggio dello stato 'step'.
        self.restore_state_clnt.call(self.step_state_name)
        # Settaggio dei nuovi parametri attraverso la action.
        # Se l'azione è il nuovo set di parametri
        if self.action_type == 'target_value':        
            for param_name in self.param_names_to_value_index.keys():
                param_value = rospy.get_param(param_name)
                if isinstance(param_value, float) or isinstance(param_value, int):
                    if (len(self.param_names_to_value_index[param_name]) == 1):
                        new_param = float(np.clip(action[self.param_names_to_value_index[param_name][0]],
                                                  self.param_lower_bound[self.param_names_to_value_index[param_name][0]],
                                                  self.param_upper_bound[self.param_names_to_value_index[param_name][0]]))
                        if new_param != new_param:
                            new_param = (self.param_upper_bound[self.param_names_to_value_index[param_name][0]] 
                                        - self.param_lower_bound[self.param_names_to_value_index[param_name][0]]) / 2
                        rospy.set_param(param_name,float(action[self.param_names_to_value_index[param_name][0]]))
                    else:
                        rospy.logerr('Current parameter and the new one have different sizes')
                elif isinstance(param_value, list):
                    if (len(self.param_names_to_value_index[param_name]) == len(param_value)):
                        values = []
                        for i in range(len(self.param_names_to_value_index[param_name])):
                            value = float(np.clip(action[self.param_names_to_value_index[param_name][i]],
                                                  self.param_lower_bound[self.param_names_to_value_index[param_name][i]],
                                                  self.param_upper_bound[self.param_names_to_value_index[param_name][i]]))
                            values.append(value)
                        rospy.set_param(param_name,values)
                    elif (len(self.param_names_to_value_index[param_name]) < len(param_value)):
                        rospy.loginfo('Current parameter has a size bigger than modification, probably some part remain equal')
                        ind = -1
                        values = []
                        for i in range(len(param_value)):
                            if i in self.param_to_avoid_index[param_name]:
                                value = param_value[i]
                            else:
                                ind += 1
                                value = float(np.clip(action[self.param_names_to_value_index[param_name][ind]],
                                    self.param_lower_bound[self.param_names_to_value_index[param_name][ind]],
                                    self.param_upper_bound[self.param_names_to_value_index[param_name][ind]]))
                            values.append(value)  
                    else:
                        rospy.logerr('Current parameter and the new one have different sizes')
                else:
                    rospy.logerr('Current param has wrong structure.')

        # Se l'azione è la variazione
        if self.action_type == 'increment_value':        
            for param_name in self.param_names_to_value_index.keys():
                param_value = rospy.get_param(param_name)
                if isinstance(param_value, float) or isinstance(param_value, int):
                    if (len(self.param_names_to_value_index[param_name]) == 1):
                        variation = action[self.param_names_to_value_index[param_name][0]]
                        if variation != variation:
                            variation = 0
                        new_param = float(np.clip(param_value + variation,
                                                  self.param_lower_bound[self.param_names_to_value_index[param_name][0]],
                                                  self.param_upper_bound[self.param_names_to_value_index[param_name][0]]))
                        rospy.set_param(param_name,new_param)
                    else:
                        rospy.logerr('Current parameter and the new one have different sizes')
                elif isinstance(param_value, list):
                    if (len(self.param_names_to_value_index[param_name]) == len(param_value)):
                        new_param = []
                        for i in range(len(param_value)):
                            variation = action[self.param_names_to_value_index[param_name][i]]
                            new_param.append(float(np.clip((param_value[i] + variation),
                                                   self.param_lower_bound[self.param_names_to_value_index[param_name][i]],
                                                   self.param_upper_bound[self.param_names_to_value_index[param_name][i]])))
                        rospy.set_param(param_name,new_param)
                    elif (len(self.param_names_to_value_index[param_name]) < len(param_value)):
                        rospy.loginfo('Current parameter has a size bigger than modification, probably some part remain equal')
                        ind = -1
                        new_param = []
                        for i in range(len(param_value)):
                            if i in self.param_to_avoid_index[param_name]:
                                new_param.append(param_value[i])
                            else:
                                ind += 1
                                variation = action[self.param_names_to_value_index[param_name][ind]]
                                new_param.append(float(np.clip((param_value[i] + variation),
                                                    self.param_lower_bound[self.param_names_to_value_index[param_name][ind]],
                                                    self.param_upper_bound[self.param_names_to_value_index[param_name][ind]])))
                        rospy.set_param(param_name,new_param)
                    else:
                        rospy.logerr('Current parameter and the new one have different sizes')
                else:
                    rospy.logerr('Current param has wrong structure.')

        # Lancio del tree. Le distanze di inizio e fine vengono registrate. 
        self._update_info
        self.initial_distance = self._distance(self.tar_pos,self.obj_pos)
        self.run_tree_clnt.call(self.tree_name, self.trees_path)
        observation = self._get_obs()
        self.final_distance = self._distance(self.tar_pos,self.obj_pos)
        
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
        self._update_info()
        dist_perc = 1 - (self.final_distance/self.initial_distance)
        rospy.loginfo('Distance percentage: ' + str(dist_perc))
        rospy.loginfo('Max wrench: [' + 
                      str(self.max_wrench[0]) + ',' + 
                      str(self.max_wrench[1]) + ',' + 
                      str(self.max_wrench[2]) + ',' + 
                      str(self.max_wrench[3]) + ',' + 
                      str(self.max_wrench[4]) + ',' + 
                      str(self.max_wrench[5]) + ']')
        reward = dist_perc * 1000000
        reward = reward + (self.max_wrench[0] * -1)
        reward = reward + (self.max_wrench[1] * -1)
        reward = reward + (self.max_wrench[3] * -1)
        reward = reward + (self.max_wrench[4] * -1)
        rospy.loginfo('Reward: ' + str(reward))
        return reward

    def _update_info(self) -> None:
        rospy.sleep(0.5)
        
        # leggo i valori dei parametri
        self.param_values.clear()
        for param_name in self.param_names_to_value_index.keys():
            param_value = rospy.get_param(param_name)
            if isinstance(param_value, int) or isinstance(param_value, float):
                self.param_values.append(param_value)
            elif isinstance(param_value, list):
                for single_value in param_value:
                    self.param_values.append(single_value)

        # leggo posizione oggetto e target
        (self.obj_pos, self.obj_rot) = self.tf_listener.lookupTransform(self.object_name, 'world', rospy.Time(0))
        (self.tar_pos, self.tar_rot) = self.tf_listener.lookupTransform(self.target_name, 'world', rospy.Time(0))

        # leggo posizioni relative di presa e rilascio 
        (self.obj_to_grasp_pos, self.obj_to_grasp_rot) = self.tf_listener.lookupTransform(self.object_name, self.object_name + '_grasp', rospy.Time(0))
        (self.tar_to_insertion_pos, self.tar_to_insertion_rot) = self.tf_listener.lookupTransform(self.target_name, self.target_name + '_insertion', rospy.Time(0))

        # leggo la forza massima del task 
        try:
            insert_executed = rospy.get_param('/exec_params/actions/can_peg_in_hole/skills/insert/executed')
            if insert_executed:
                try:
                    self.max_wrench = rospy.get_param('/exec_params/actions/can_peg_in_hole/skills/insert/max_wrench')
                except:
                    rospy.logerr('/exec_params/actions/can_peg_in_hole/skills/insert/max_wrench not found')
                    self.max_wrench = [100000,100000,100000,100000,100000,100000]                
            else:
                self.max_wrench = [100000,100000,100000,100000,100000,100000]
        except:
            rospy.logwarn('/exec_params/actions/can_peg_in_hole/skills/insert/executed not found, it considered false')
            self.max_wrench = [100000,100000,100000,100000,100000,100000]

    def _print_action(self, action) -> None:
        print('ACTION____________________________________________________________________')
        for param_name in self.param_names_to_value_index.keys():
            for index in self.param_names_to_value_index[param_name]:
                print(param_name + str(index) + ': ' + str(action[index]))
        print('__________________________________________________________________________')

    def _print_obs(self, observation) -> None:
        print('OBSERVATION_______________________________________________________________')
        for param_name in self.param_names_to_value_index.keys():
            for index in self.param_names_to_value_index[param_name]:
                print(param_name + str(index) + ': ' + str(self.param_values[index]))
        
        print('obj_to_grasp_pos: ' + str(observation[len(self.param_values):len(self.param_values)+3]))
        print('tar_to_insertion_pos: ' + str(observation[len(self.param_values)+4:len(self.param_values)+7]))
        print('max_wrench: ' + str(observation[len(self.param_values)+8:len(self.param_values)+14]))
        print('__________________________________________________________________________')
