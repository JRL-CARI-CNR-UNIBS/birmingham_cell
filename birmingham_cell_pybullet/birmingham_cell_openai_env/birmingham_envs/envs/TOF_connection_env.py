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

import copy

class TOFConnectionEnv(gym.Env):
    
    def __init__(
        self,
        node_name: str = 'Connection_env',
        package_name: str = 'birmingham_cell_tests',
        trees_path: str = '/config/trees',
        tree_name: str = 'tof_can_peg_in_hole',
        object_name: str = 'can',
        target_name: str = 'hole',
        obj_model_name: str = 'can',
        tar_model_name: str = 'hole',
        obj_model_height: float = 0.1,
        obj_model_length: float = 0.1,
        obj_model_width:  float = 0.1,
        tar_model_height: float = 0.1,
        tar_model_length: float = 0.1,
        tar_model_width:  float = 0.1,
        distance_threshold: float = 0.01,
        force_threshold: float = 50,
        torque_threshold: float = 100,
        action_type: str = 'increment_value',
        randomized_tf: list = ['can_grasp', 'hole_insertion'],
        debug_mode: bool = False,
        start_epoch_number: int = 0,
        data_file_name: str = 'td3_tests',
        save_data: bool = False,
        step_print: bool = False,
        only_pos_success: bool = True,
        observation_type: str = 'param_pos_obj',
        epoch_len: int = None,
    ) -> None:
        rospy.init_node(node_name)

        rospack = rospkg.RosPack()
        self.package_path = rospack.get_path(package_name)
        self.trees_path = [self.package_path + trees_path]
        self.tree_name  = tree_name
        self.object_name      = object_name
        self.target_name      = target_name
        self.obj_model_name   = obj_model_name
        self.tar_model_name   = tar_model_name
        self.obj_model_height = obj_model_height
        self.obj_model_length = obj_model_length
        self.obj_model_width  = obj_model_width
        self.tar_model_height = tar_model_height
        self.tar_model_length = tar_model_length
        self.tar_model_width  = tar_model_width
        self.distance_threshold = distance_threshold
        self.force_threshold    = force_threshold
        self.torque_threshold   = torque_threshold
        self.action_type = action_type   # currently available, target_value  increment_value  
        self.randomized_tf = randomized_tf
        self.debug_mode = debug_mode
        self.epoch_number = start_epoch_number
        self.data_file_name = data_file_name + '.ods'
        self.save_data = save_data
        self.step_print = step_print
        self.only_pos_success = only_pos_success
        self.obs_type = observation_type
        self.step_number = 0
        self.start_obj_pos = None
        self.start_tar_pos = None
        self.epoch_len = epoch_len
        self.epoch_steps = 0
        self.grasp_zone = 0 
        
        # arguments to define
        self.param_lower_bound = []
        self.param_upper_bound = []
        self.param_names_to_value_index = {}
        self.param_to_avoid_index = {}
        self.param_values = []
        self.initial_param_values = {}
        self.obj_to_grasp_pos = []
        self.obj_to_grasp_rot = []
        self.tar_to_insertion_pos = []
        self.tar_to_insertion_rot = []
        self.initial_distance = None
        self.final_distance = None
        self.all_param_names = []
        self.param_history = []
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
        self.default_grasp_pos = None
        self.default_insert_pos = None
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
        self.observation, _ = self.reset()  # required for init; seed can be changed later
        rospy.loginfo("Reset done")
        observation_shape = self.observation.shape
        self.observation_space = spaces.Box(-1, 1, shape=observation_shape, dtype=np.float64)

        # Definizione dell'action space
        self.action_space = spaces.Box(-1,1, shape=(1,), dtype=np.float64)

    def _create_scene(self,obj_pos,tar_pos) -> None:
        """Create the scene."""

        if self.tar_model_name == 'box_hole':
            rospy.set_param('pybullet_simulation/objects/box_hole/xacro_args/length', self.tar_model_length)
            rospy.set_param('pybullet_simulation/objects/box_hole/xacro_args/width',  self.tar_model_width )
            rospy.set_param('pybullet_simulation/objects/box_hole/xacro_args/height', self.tar_model_height)
        elif self.tar_model_name == 'cylinder_hole':
            rospy.set_param('pybullet_simulation/objects/cylinder_hole/xacro_args/radius', (self.tar_model_width / 2) )
            rospy.set_param('pybullet_simulation/objects/cylinder_hole/xacro_args/height', self.tar_model_height)

        object_names = []
        object_names.append(self.target_name)  
        object_names.append(self.object_name)
        self.delete_model_clnt.call(object_names)

        model_name = []
        pose = []
        fixed = []
        tar_pose = Pose()
        self.start_tar_pos = tar_pose

        tar_pose.position.x = tar_pos[0]
        tar_pose.position.y = tar_pos[1]
        tar_pose.position.z = tar_pos[2]
        tar_pose.orientation.x = 0.0
        tar_pose.orientation.y = 0.0
        tar_pose.orientation.z = 0.0
        tar_pose.orientation.w = 1.0
        model_name.append(self.tar_model_name)
        pose.append(tar_pose)
        fixed.append(True)

        self.start_obj_pos = obj_pos

        if self.obj_model_name == 'box':
            rospy.set_param('pybullet_simulation/objects/box/xacro_args/length', self.obj_model_length)
            rospy.set_param('pybullet_simulation/objects/box/xacro_args/width',  self.obj_model_width )
            rospy.set_param('pybullet_simulation/objects/box/xacro_args/height', self.obj_model_height)
        elif self.obj_model_name == 'cylinder':
            rospy.set_param('pybullet_simulation/objects/cylinder/xacro_args/radius', (self.obj_model_width / 2) )
            rospy.set_param('pybullet_simulation/objects/cylinder/xacro_args/height', self.obj_model_height)
        elif self.obj_model_name == 'sphere':
            rospy.set_param('pybullet_simulation/objects/sphere/xacro_args/radius', (self.obj_model_height / 2) )

        obj_pose = Pose()
        obj_pose.position.x = obj_pos[0]
        obj_pose.position.y = obj_pos[1]
        obj_pose.position.z = obj_pos[2]
        obj_pose.orientation.x = 0.0
        obj_pose.orientation.y = 0.0
        obj_pose.orientation.z = 0.0
        obj_pose.orientation.w = 1.0
        model_name.append(self.obj_model_name)
        pose.append(obj_pose)
        fixed.append(False)
        self.spawn_model_clnt.call(object_names, model_name, pose, fixed)

        # randomizzo le posizioni di grasp e di insertion
        new_tfs = []
        for old_tf in self.initial_tf:
            new_tf = copy.copy(old_tf)
            if new_tf['name'] in self.randomized_tf:
                low_limit = [-0.02, -0.02, 0.0]
                high_limit = [0.02, 0.02, 0.02]
                noise = np.ndarray.tolist(self.np_random.uniform(low_limit, high_limit))
                new_tf['position'] = np.ndarray.tolist(np.add(new_tf['position'], noise))
                if new_tf['name'] == 'can_grasp':
                    self.default_grasp_pos = copy.copy(new_tf['position'])
                    self.current_grasp_pos = copy.copy(new_tf['position'])
                if new_tf['name'] == 'hole_insertion':
                    self.default_insert_pos = copy.copy(new_tf['position'])
                    self.current_insert_pos = copy.copy(new_tf['position'])
            new_tfs.append(new_tf)
        rospy.set_param('tf_params',new_tfs)
        rospy.sleep(0.5)

    def _get_obs(self) -> Dict[str, np.array]:
        # Come osservazione utilizzo le posizioni relative e il set di parametri
        self._update_info()
        if self.obs_type == 'param_pos':
            self.observation = np.concatenate([np.array(self.param_values),np.array(self.current_grasp_pos),np.array(self.current_insert_pos)])
        elif self.obs_type == 'param_pos_obj':
            self.observation = np.concatenate([np.array(self.param_values),np.array(self.current_grasp_pos),np.array(self.current_insert_pos),np.array(self.object_info)])
        elif self.obs_type == 'param_pos_zone_force_obj':
            self.observation = np.concatenate([np.array(self.param_values),np.array(self.current_grasp_pos),np.array(self.current_insert_pos),np.array([self.grasp_zone]),np.array(self.grasp_max_wrench[:2]),np.array(self.insertion_max_wrench[:2]),np.array(self.object_info)])
        else:
            rospy.logerr('Observation type param_pos_zone_force_obj does not exist')
            exit(0)
        if self.debug_mode: self._print_obs()
        return self.observation

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[Dict[str, np.array], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        
        self.epoch_steps = 0

        if self.step_print:
            print('epoch ' + self.epoch_number)
        # salvo i dati ottenuti dal'epoca precedente 
        if self.save_data:
            if self.param_history:
                self.epoch_number += 1
                data = []
                data.append(self.all_param_names + ['reward','iteration'])
                for index in range(len(self.param_history)):
                    data.append(self.param_history[index])
                param_history_ods = od.get_data(self.package_path + "/data/td3_tests.ods")
                param_history_ods.update({str(self.epoch_number): data})
                od.save_data(self.package_path + "/data/" + self.data_file_name, param_history_ods)
                self.param_history.clear()
                self.step_number = 0
            else:
                rospy.logwarn('Nothing to save')

        # rimuovo gli oggetti della scena, riporto il robot nello stato iniziale e poi riaggiungo gli oggetti 
        # in una posizione casuale ma non sovrapposta
        object_names = [self.target_name, self.object_name]
        print('Wait to delete objects')
        self.delete_model_clnt.call(object_names)
        print('Objects deleted')
        print('Wait to delete step state')
        self.delete_state_clnt.call([self.step_state_name])
        print('State deleted')
        rospy.loginfo("Model and step_state deleted")

        self.restore_state_clnt.call(self.reset_state_name)
        rospy.loginfo("Reset_state restored")

        # Setto i parametri come nello stato iniziale così da ripartire da zero nella modifica dei parametri 
        for param_name in self.initial_param_values.keys():
            rospy.set_param(param_name,self.initial_param_values[param_name])


        if self.obj_model_name == 'cylinder':
            self.object_type = 0
        elif self.obj_model_name == 'box':
            self.object_type = 1
        elif self.obj_model_name == 'cone':
            self.object_type = 2
        elif self.obj_model_name == 'sphere':
            self.object_type = 3
        elif self.obj_model_name == 'can':
            self.object_type = 0
            self.obj_model_height = 0.11
        self.object_info = [self.object_type,self.obj_model_height]


        self.start_tar_pos = self._sample_target()
        self.start_obj_pos = self._sample_object()
        self._create_scene(self.start_obj_pos,self.start_tar_pos)
        rospy.loginfo("Scene created")

        self.save_state_clnt(self.step_state_name)

        self.observation = self._get_obs()
        info = {"is_success": False}
        return self.observation, info
    
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
            if self.obj_model_name == 'can':
                object_position = np.array([0.0, 0.0, 0.0])
            else :
                object_position = np.array([0.0, 0.0, self.obj_model_height / 2])

            noise = self.np_random.uniform(self.work_space_range_low, self.work_space_range_high)
            object_position += noise
            if (self._distance(self.start_tar_pos[0:2],object_position[0:2]) > 0.25):
                finish = True
        return object_position

    def _is_success(self) -> np.array:
        self._update_info()
        if self.only_pos_success:
            print('Distance threshold: ' + str(self.distance_threshold))
            if (self.final_distance < self.distance_threshold):
                success = True
                print('True')
            else:
                success = False
                print('False')
        else:
            if (self.final_distance < self.distance_threshold and
                self.insertion_max_wrench[0] < self.force_threshold and
                self.insertion_max_wrench[1] < self.force_threshold and
                self.insertion_max_wrench[3] < self.torque_threshold and
                self.insertion_max_wrench[4] < self.torque_threshold):
                success = True
            else:
                success = False
        return np.array(success, dtype=bool)

    def step(self, action: np.array) -> Tuple[Dict[str, np.array], float, bool, bool, Dict[str, Any]]:
        self.epoch_steps += 1
        self.step_number += 1
        
        self.restore_state_clnt.call(self.step_state_name)

        # Lancio del tree. Le distanze di inizio e fine vengono registrate. 
        self._update_info()
        self.initial_distance = self._distance(self.tar_pos,self.obj_pos)
        self.initial_obj_to_grasp_pos = self.obj_to_grasp_pos
        self.run_tree_clnt.call(self.tree_name, self.trees_path)
        self._get_obs()
        self.final_distance = self._distance(self.tar_pos,self.obj_pos)
        print('tar_pos: ' + str(self.tar_pos))
        print('obj_pos: ' + str(self.obj_pos))
        success = bool(self._is_success())

        if ((self.epoch_len is not None) and success):
            single_reward = self._get_reward()
            remain_step = self.epoch_len - self.epoch_steps
            reward = single_reward + remain_step
        else:
            reward = self._get_reward()
        
        self._get_obs()
        terminated = success
        truncated = False
        info = {"is_success": success}
        
        rospy.set_param('/exec_params/actions/can_peg_in_hole/total_reward', float(reward))
        # print(self.obj_to_grasp_pos)
        # print(self.tar_to_insertion_pos)
        return self.observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.array]:
        return 

    def _distance(self, pos1: np.array, pos2: np.array) -> float:
        return np.linalg.norm(np.array(pos1)-np.array(pos2))
    
    def _get_reward(self) -> float:
        self._update_info()
        dist_perc = 1 - (self.final_distance/self.initial_distance)
        try:
            move_to_grasp_fail = rospy.get_param('/exec_params/actions/can_peg_in_hole/skills/move_to_can_grasp/fail')
        except:
            rospy.logwarn('/exec_params/actions/can_peg_in_hole/skills/move_to_can_grasp/fail not found, it considered true')
            move_to_grasp_fail = True

        try:
            move_to_grasp_contant = rospy.get_param('/exec_params/actions/can_peg_in_hole/skills/move_to_can_grasp/contact')
        except:
            rospy.logwarn('/exec_params/actions/can_peg_in_hole/skills/move_to_can_grasp/fail not found, it considered true')
            move_to_grasp_contant = False

        grasp_zone = str()

        if move_to_grasp_fail:
            grasp_zone = 'collision'
            self.grasp_zone = 1 
        elif move_to_grasp_contant:
            grasp_zone = 'out_grasp'
            self.grasp_zone = 0
        elif (dist_perc > 0.5):
            grasp_zone = 'in_grasp'
            self.grasp_zone = 2
        else:
            grasp_zone = 'out_grasp'
            self.grasp_zone = 0

        if (grasp_zone == 'in_grasp'):
            print('In grasp')
            if self.only_pos_success:
                missing_dist_perc = 1 - dist_perc # max value 0.5
                reward = 1 
                reward -= missing_dist_perc * 1.4 # max value 0.7
                # print('initial_distance:   ' + str(self.initial_distance))
                # print('final_distance:     ' + str(self.final_distance))
                # print('Dist_perc:          ' + str(dist_perc))
                # print('missing_dist_perc:  ' + str(missing_dist_perc))
            else:
                missing_dist_perc = 1 - dist_perc # max value 0.5
                self.insertion_max_wrench[0] # max 1000 
                self.insertion_max_wrench[1] # max 1000
                self.insertion_max_wrench[3] # max 1000
                self.insertion_max_wrench[4] # max 1000   
                reward = 1
                reward -= missing_dist_perc # max 0.5
                reward -= self.insertion_max_wrench[0] * 0.00005 # max 0.05
                reward -= self.insertion_max_wrench[1] * 0.00005 # max 0.05
                reward -= self.insertion_max_wrench[3] * 0.00005 # max 0.05
                reward -= self.insertion_max_wrench[4] * 0.00005 # max 0.05
        elif (grasp_zone == 'collision'):
            # print('In collision')
            dist_xy_grasp = np.linalg.norm(self.current_grasp_pos[0:2])
            dist_flor_to_grasp = self.gripper_grasp_pos[2]
            reward = 0
            reward += dist_flor_to_grasp # max 0.1. 
            reward -= dist_xy_grasp # max 0.1. 
            # print('current_grasp_pos:  ' + str(self.current_grasp_pos[0:2]))
            # print('dist_xy_grasp:      ' + str(dist_xy_grasp))
            # print('dist_flor_to_grasp: ' + str(self.gripper_grasp_pos[2]))
        else:
            # print('Free')
            dist_grasp = np.linalg.norm(self.current_grasp_pos)
            reward = -0.3
            reward -= dist_grasp
            # print('current_grasp_pos:  ' + str(self.current_grasp_pos))

        # print('Reward:             '+ str(reward))
        rospy.set_param('/exec_params/actions/can_peg_in_hole/skills/insert/executed',False)

        # Qua riempio lo storico dei parametri e il relativo reward
        # if self.save_data:
        #     self.param_history.append(self.param_values + [float(reward),self.step_number] + self.last_action + self.observation)

        return reward

    def _update_info(self) -> None:        
        # leggo i valori dei parametri
        self.param_values.clear()

        for param_name in self.param_names_to_value_index.keys():
            param_value = rospy.get_param(param_name)
            if isinstance(param_value, int) or isinstance(param_value, float):
                self.param_values.append(param_value)
            elif isinstance(param_value, list):
                if (len(self.param_names_to_value_index[param_name]) == len(param_value)):
                    for single_value in param_value:
                        self.param_values.append(single_value)
                elif (len(self.param_names_to_value_index[param_name]) < len(param_value)):
                    for i in range(len(param_value)):
                        if not i in self.param_to_avoid_index[param_name]:
                            self.param_values.append(param_value[i])

        # print(self.current_grasp_pos)
        # print(self.default_grasp_pos)
        # print(self.param_values)

        self.current_grasp_pos[0] = self.default_grasp_pos[0] + self.param_values[0]
        self.current_grasp_pos[1] = self.default_grasp_pos[1] - self.param_values[1]
        self.current_grasp_pos[2] = self.default_grasp_pos[2] - self.param_values[2]
        self.current_insert_pos[0] = self.default_insert_pos[0] + self.param_values[3]
        self.current_insert_pos[1] = self.default_insert_pos[1] - self.param_values[4]
        self.current_insert_pos[2] = self.default_insert_pos[2] - self.param_values[5]

        if not self.obj_pos is None:
            self.old_obj_pos = self.obj_pos
            self.old_obj_rot = self.obj_rot
            self.old_tar_pos = self.tar_pos
            self.old_tar_rot = self.tar_rot
        # leggo posizione oggetto e target
        (self.obj_pos, self.obj_rot) = self.tf_listener.lookupTransform(self.object_name, 'world', rospy.Time(0))
        (self.tar_pos, self.tar_rot) = self.tf_listener.lookupTransform(self.target_name, 'world', rospy.Time(0))

        # leggo posizioni relative di presa e rilascio 
        try:
            (self.obj_to_grasp_pos, self.obj_to_grasp_rot) = self.tf_listener.lookupTransform(self.object_name, self.object_name + '_grasp_goal', rospy.Time(0))
        except:
            (self.obj_to_grasp_pos, self.obj_to_grasp_rot) = self.tf_listener.lookupTransform(self.object_name, self.object_name + '_grasp', rospy.Time(0))
        try:
            (self.tar_to_insertion_pos, self.tar_to_insertion_rot) = self.tf_listener.lookupTransform(self.target_name, self.target_name + '_insertion_goal', rospy.Time(0))
        except:    
            (self.tar_to_insertion_pos, self.tar_to_insertion_rot) = self.tf_listener.lookupTransform(self.target_name, self.target_name + '_insertion', rospy.Time(0))

        try:
            (self.gripper_grasp_pos, self.gripper_grasp_rot) = self.tf_listener.lookupTransform('world', self.object_name + '_grasp_goal', rospy.Time(0))
        except:
            (self.gripper_grasp_pos, self.gripper_grasp_rot) = self.tf_listener.lookupTransform('world', self.object_name + '_grasp', rospy.Time(0))

        if self.debug_mode:
            print('current_grasp_pos: ' +str(self.current_grasp_pos))
            print('current_insert_pos: ' +str(self.current_insert_pos))
            print('obj_to_grasp_pos: ' +str(self.obj_to_grasp_pos))
            print('tar_to_insertion_pos: ' +str(self.tar_to_insertion_pos))

        # leggo la forza massima del task 
        m_w_p_value = 1000
        try:
            insert_executed = rospy.get_param('/exec_params/actions/can_peg_in_hole/skills/insert/executed')
            if insert_executed:
                try:
                    self.insertion_max_wrench = rospy.get_param('/exec_params/actions/can_peg_in_hole/skills/insert/max_wrench')
                except:
                    rospy.logerr('/exec_params/actions/can_peg_in_hole/skills/insert/max_wrench not found')
                    self.insertion_max_wrench = [m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value]                
            else:
                self.insertion_max_wrench = [m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value]
        except:
            rospy.logwarn('/exec_params/actions/can_peg_in_hole/skills/insert/executed not found, it considered false')
            self.insertion_max_wrench = [m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value]

        try:
            grasp_executed = rospy.get_param('/exec_params/actions/can_peg_in_hole/skills/insert/executed')
            if grasp_executed:
                try:
                    self.grasp_max_wrench = rospy.get_param('/exec_params/actions/can_peg_in_hole/skills/close_gripper/max_wrench')
                except:
                    rospy.logerr('/exec_params/actions/can_peg_in_hole/skills/close_gripper/max_wrench not found')
                    self.grasp_max_wrench = [m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value]                
            else:
                self.grasp_max_wrench = [m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value]
        except:
            rospy.logwarn('/exec_params/actions/can_peg_in_hole/skills/close_gripper/executed not found, it considered false')
            self.grasp_max_wrench = [m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value,m_w_p_value]

        if self.grasp_zone != 2:
            self.grasp_max_wrench = [0,0,0,0,0,0]
            self.insertion_max_wrench = [0,0,0,0,0,0]

    def _print_obs(self) -> None:
        print(' ')
        print('OBSERVATION_______________________________________________________________')
        for param_name in self.param_names_to_value_index.keys():
            if len(self.param_names_to_value_index[param_name]) == 1:
                print(param_name + ': ' + str(self.observation[self.param_names_to_value_index[param_name][0]]))
            else:
                param_len = len(self.param_names_to_value_index[param_name])
                start_index = self.param_names_to_value_index[param_name][0]
                print(param_name + ': ' + str(self.observation[start_index:start_index+param_len]))

        start_index = len(self.param_values)
        print('obj_to_grasp_pos: ' + str(self.observation[start_index:start_index+3]))
        print('tar_to_insertion_pos: ' + str(self.observation[start_index+3:start_index+6]))
        # print('max_wrench: ' + str(self.observation[start_index+6:start_index+12]))
        print('--------------------------------------------------------------------------')
        print(' ')

    def clean_scene(self) -> None:
        """Create the scene."""
        object_names = []
        object_names.append(self.target_name)  
        object_names.append(self.object_name)
        self.delete_model_clnt.call(object_names)