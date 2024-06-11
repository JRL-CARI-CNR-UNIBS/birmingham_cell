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
from geometry_msgs.msg import Pose, WrenchStamped

import copy

class ConnectionForcesEnv(gym.Env):
    
    def __init__(
        self,
        node_name: str = 'ConnectionForcesEnv',
        package_name: str = 'birmingham_cell_tests',

        tar_model_name: str = 'cylinder_hole',
        tar_model_height: float = 0.07,
        tar_model_length: float = 0.1,
        tar_model_width:  float = 0.1,
        tar_model_radius: float = 0.032,
        obj_model_name: str = 'cylinder',
        obj_model_height: float = 0.1,
        obj_model_length: float = 0.1,
        obj_model_width:  float = 0.1,
        obj_model_radius: float = 0.03,
        object_name: str = 'can',
        target_name: str = 'hole',
        distance_threshold: float = 0.01,
        epoch_len: int = 1,
    ) -> None:
        rospy.init_node(node_name)

        self.object_name = object_name
        self.target_name = target_name

        # setto i parametri degli oggetti
        if obj_model_name == 'cylinder':
            rospy.set_param('pybullet_simulation/objects/cylinder/xacro_args/height', obj_model_height)
            rospy.set_param('pybullet_simulation/objects/cylinder/xacro_args/radius', obj_model_radius)
            self.grasp_height = (obj_model_height / 2) - 0.015
        elif obj_model_name == 'box':
            rospy.set_param('pybullet_simulation/objects/box/xacro_args/length', obj_model_length)
            rospy.set_param('pybullet_simulation/objects/box/xacro_args/width', obj_model_width)
            rospy.set_param('pybullet_simulation/objects/box/xacro_args/height', obj_model_height)
            self.grasp_height = (obj_model_height / 2) - 0.015
        if tar_model_name == 'cylinder_hole':
            rospy.set_param('pybullet_simulation/objects/cylinder_hole/xacro_args/height', tar_model_height)
            rospy.set_param('pybullet_simulation/objects/cylinder_hole/xacro_args/radius', tar_model_radius)
        elif tar_model_name == 'box_hole':
            rospy.set_param('pybullet_simulation/objects/box_hole/xacro_args/length', tar_model_length)
            rospy.set_param('pybullet_simulation/objects/box_hole/xacro_args/width', tar_model_width)
            rospy.set_param('pybullet_simulation/objects/box_hole/xacro_args/height', tar_model_height)


        self.tf_listener = tf.TransformListener()

        rospack = rospkg.RosPack()
        pack_path = rospack.get_path(package_name)
        self.trees_path = [pack_path + '/config/trees',
                           pack_path + '/config/trees/collecting_data']
        
        self.grasp_param_name = '/exec_params/actions/grasping/skills/move_to_grasp/relative_position'
        self.insert_param_name = '/exec_params/actions/can_peg_in_hole/skills/move_to_insertion/relative_position'

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

        self.recording = False
        sub = rospy.Subscriber('/panda/panda_hand_joint/wrench',WrenchStamped, self._read_wrench_cb)

        # Setto il robot e scena nella sua posizione iniziale e salvo lo stato pybullet
        self.run_tree_clnt.call('init_tree', self.trees_path)

        object_names = []
        object_names.append(target_name)
        object_names.append(object_name)
        self.delete_model_clnt.call(object_names)

        model_name = []
        pose = []
        fixed = []

        tar_pose = Pose()
        tar_pose.position.x = 0.6
        tar_pose.position.y = 0.25
        tar_pose.position.z = 0
        tar_pose.orientation.x = 0.0
        tar_pose.orientation.y = 0.0
        tar_pose.orientation.z = 0.0
        tar_pose.orientation.w = 1.0
        model_name.append(tar_model_name)
        pose.append(tar_pose)
        fixed.append(True)

        obj_pose = Pose()
        obj_pose.position.x = 0.6
        obj_pose.position.y = 0
        obj_pose.position.z = 0.06
        obj_pose.orientation.x = 0.0
        obj_pose.orientation.y = 0.0
        obj_pose.orientation.z = 0.0
        obj_pose.orientation.w = 1.0
        model_name.append(obj_model_name)
        pose.append(obj_pose)
        fixed.append(False)
        self.spawn_model_clnt.call(object_names, model_name, pose, fixed)

        self.save_state_clnt.call('reset')
        rospy.loginfo("Init done")

        self.distance_threshold = distance_threshold
        self.step_number = 0
        self.epoch_len = epoch_len
        self.epoch_steps = 0
               
        self.param_upper_bound = np.array([ 0.05, 0.05, 0.05, 0.05])
        self.param_lower_bound = np.array([-0.05,-0.05,-0.05,-0.05])
        self.init_par_val      = np.array([ 0.00, 0.00, 0.00, 0.00])

        self.wrench_record = {}
        self.wrench_record['fx'] = np.array([])
        self.wrench_record['fy'] = np.array([])
        self.wrench_record['fz'] = np.array([])
        self.wrench_record['tx'] = np.array([])
        self.wrench_record['ty'] = np.array([])
        self.wrench_record['tz'] = np.array([])
        sampling_freq = 250
        grasp_max_time = 1
        insert_max_time = 2
        self.grasp_recording_lenght = grasp_max_time * sampling_freq
        self.insert_recording_lenght = insert_max_time * sampling_freq
        # chiamo il reset, in questo genero l'errore nelle posizioni (tf) e lancio la prima esecuzione
        observation, _ = self.reset()  # required for init; seed can be changed later
        rospy.loginfo("Reset done")
        observation_shape = observation.shape
        self.observation_space = spaces.Box(-1, 1, shape=observation_shape, dtype=np.float64)

        space_division = 10.0
        self.max_variations = (self.param_upper_bound - self.param_lower_bound)/space_division
        self.action_space = spaces.Box(-1, 1, shape=(len(self.max_variations),), dtype=np.float64)

    def _get_obs(self) -> Dict[str, np.array]:
        observation = np.concatenate([self.param_values,self.current_grasp_pos,self.current_insert_pos,self.grasp_forces/10000,self.insert_forces/10000])
        return observation

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[Dict[str, np.array], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        
        self.epoch_steps = 0

        self.restore_state_clnt.call('reset')
        rospy.loginfo("Reset_state restored")

        # Setto i parametri come nello stato iniziale così da ripartire da zero nella modifica dei parametri 
        self.param_values = copy.copy(self.init_par_val)
        rospy.set_param(self.grasp_param_name, [float(self.param_values[0]),float(self.param_values[1]),0.0])
        rospy.set_param(self.insert_param_name, [float(self.param_values[2]),float(self.param_values[3]),0.0])

        # genero errore nelle pos di presa e inserimento e ne modifico le TF
        grasp_pose_err_low_limit = [-0.01, -0.01]
        grasp_pose_err_high_limit = [0.01, 0.01]
        insert_pose_err_low_limit = [-0.01, -0.01]
        insert_pose_err_high_limit = [0.01, 0.01]
        obj_pos_error = [0.0,0.0]
        while np.linalg.norm(obj_pos_error) < 0.01:
            obj_pos_error = np.ndarray.tolist(self.np_random.uniform(grasp_pose_err_low_limit, grasp_pose_err_high_limit))
        tar_pos_error = [0.0,0.0]
        while np.linalg.norm(tar_pos_error) < 0.01:
            tar_pos_error = np.ndarray.tolist(self.np_random.uniform(insert_pose_err_low_limit, insert_pose_err_high_limit))
        current_tfs = rospy.get_param('tf_params')
        new_tfs = []
        for tf in current_tfs:
            new_tf = copy.copy(tf)
            if new_tf['name'] == 'hole_insertion':
                new_tf['position'][0] = tar_pos_error[0]
                new_tf['position'][1] = tar_pos_error[1]
                self.initial_insert_pos = new_tf['position'][0:2]
            if new_tf['name'] == 'can_grasp':
                new_tf['position'][0] = obj_pos_error[0]
                new_tf['position'][1] = obj_pos_error[1]
                new_tf['position'][2] = self.grasp_height
                self.initial_grasp_pos = new_tf['position'][0:2]
            new_tfs.append(new_tf)
        rospy.set_param('tf_params',new_tfs)
        
        self.current_insert_pos  = copy.copy(self.initial_insert_pos)
        self.current_grasp_pos  = copy.copy(self.initial_grasp_pos)
        
        self._run_task()

        observation = self._get_obs()
        info = {"is_success": False}
        return observation, info
    
    def _is_success(self) -> np.array:
        (self.obj_pos, obj_rot) = self.tf_listener.lookupTransform(self.object_name, 'world', rospy.Time(0))
        (self.tar_pos, tar_rot) = self.tf_listener.lookupTransform(self.target_name, 'world', rospy.Time(0))

        distance = self._distance(self.obj_pos, self.tar_pos)

        if distance < self.distance_threshold:
            success = True
        else:
            success = False
        return np.array(success, dtype=bool)

    def step(self, action: np.array) -> Tuple[Dict[str, np.array], float, bool, bool, Dict[str, Any]]:
        self.last_action = action.tolist()
        self.epoch_steps += 1

        self.restore_state_clnt.call('reset')

        self.param_values = np.add(self.param_values, np.multiply(action, self.max_variations))
        self.param_values = np.clip(self.param_values, self.param_lower_bound, self.param_upper_bound)
        rospy.set_param(self.grasp_param_name, [float(self.param_values[0]),float(self.param_values[1]),0.0])
        rospy.set_param(self.insert_param_name, [float(self.param_values[2]),float(self.param_values[3]),0.0])
        self.current_grasp_pos[0] = self.initial_grasp_pos[0] + self.param_values[0]
        self.current_grasp_pos[1] = self.initial_grasp_pos[1] - self.param_values[1]
        self.current_insert_pos[0] = self.initial_insert_pos[0] + self.param_values[2]
        self.current_insert_pos[1] = self.initial_insert_pos[1] - self.param_values[3]

        self._run_task()

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
        info = {"is_success": success}
        
        return observation, reward, terminated, truncated, info

    def _distance(self, pos1: np.array, pos2: np.array) -> float:
        return np.linalg.norm(np.array(pos1)-np.array(pos2))
    
    def _get_reward(self) -> float:
        reward = 1 - self._distance(self.obj_pos, self.tar_pos)       
        return reward

    def _read_wrench_cb(self, data):
        if self.recording:
            self.wrench_record['fx'].append(data.wrench.force.x)
            self.wrench_record['fy'].append(data.wrench.force.y)
            self.wrench_record['fz'].append(data.wrench.force.z)
            self.wrench_record['tx'].append(data.wrench.torque.x)
            self.wrench_record['ty'].append(data.wrench.torque.y)
            self.wrench_record['tz'].append(data.wrench.torque.z)

    def _pad_forces(self, vector, lenght):
        return np.array(np.ndarray.tolist(vector) + [0.0] * (lenght - len(vector)))

    def _run_task(self):
        result = self.run_tree_clnt.call('to_grasping', self.trees_path)

        if result.result < 3:
            self.wrench_record = {}
            self.wrench_record['fx'] = []
            self.wrench_record['fy'] = []
            self.wrench_record['fz'] = []
            self.wrench_record['tx'] = []
            self.wrench_record['ty'] = []
            self.wrench_record['tz'] = []
            self.recording = True
            self.run_tree_clnt.call('grasping', self.trees_path)
            self.recording = False

            self.wrench_record['fx'] = self._pad_forces(np.array(self.wrench_record['fx'][::6]), self.grasp_recording_lenght)
            self.wrench_record['fy'] = self._pad_forces(np.array(self.wrench_record['fy'][::6]), self.grasp_recording_lenght)
            self.wrench_record['fz'] = self._pad_forces(np.array(self.wrench_record['fz'][::6]), self.grasp_recording_lenght)
            self.wrench_record['tx'] = self._pad_forces(np.array(self.wrench_record['tx'][::6]), self.grasp_recording_lenght)
            self.wrench_record['ty'] = self._pad_forces(np.array(self.wrench_record['ty'][::6]), self.grasp_recording_lenght)
            self.wrench_record['tz'] = self._pad_forces(np.array(self.wrench_record['tz'][::6]), self.grasp_recording_lenght)
            self.grasp_forces = np.concatenate([self.wrench_record['fx'],
                                                self.wrench_record['fy'],
                                                self.wrench_record['fz'],
                                                self.wrench_record['tx'],
                                                self.wrench_record['ty'],
                                                self.wrench_record['tz'],])

            result = self.run_tree_clnt.call('move_to_insertion', self.trees_path)
            if result.result < 3:
                self.wrench_record = {}
                self.wrench_record['fx'] = []
                self.wrench_record['fy'] = []
                self.wrench_record['fz'] = []
                self.wrench_record['tx'] = []
                self.wrench_record['ty'] = []
                self.wrench_record['tz'] = []

                self.recording = True
                self.run_tree_clnt.call('insertion', self.trees_path)
                self.recording = False

                self.wrench_record['fx'] = self._pad_forces(np.array(self.wrench_record['fx'][::6]), self.insert_recording_lenght)
                self.wrench_record['fy'] = self._pad_forces(np.array(self.wrench_record['fy'][::6]), self.insert_recording_lenght)
                self.wrench_record['fz'] = self._pad_forces(np.array(self.wrench_record['fz'][::6]), self.insert_recording_lenght)
                self.wrench_record['tx'] = self._pad_forces(np.array(self.wrench_record['tx'][::6]), self.insert_recording_lenght)
                self.wrench_record['ty'] = self._pad_forces(np.array(self.wrench_record['ty'][::6]), self.insert_recording_lenght)
                self.wrench_record['tz'] = self._pad_forces(np.array(self.wrench_record['tz'][::6]), self.insert_recording_lenght)
                self.insert_forces = np.concatenate([self.wrench_record['fx'],
                                                     self.wrench_record['fy'],
                                                     self.wrench_record['fz'],
                                                     self.wrench_record['tx'],
                                                     self.wrench_record['ty'],
                                                     self.wrench_record['tz'],])
            else:
                self.insert_forces = np.array([0.0,])
                self.insert_forces = self._pad_forces(self.insert_forces, self.insert_recording_lenght * 6)
        else:
            self.grasp_forces = np.array([0.0,])
            self.grasp_forces = self._pad_forces(self.grasp_forces, self.grasp_recording_lenght * 6)
            self.insert_forces = np.array([0.0,])
            self.insert_forces = self._pad_forces(self.insert_forces, self.insert_recording_lenght * 6)
