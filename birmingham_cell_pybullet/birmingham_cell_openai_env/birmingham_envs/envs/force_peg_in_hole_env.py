#!/usr/bin/env python3
from inspect import TPFLAGS_IS_ABSTRACT
from typing import Any, Dict, Optional, Tuple

import rospkg
import rospy
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import ast


class ForcePegInHoleEnv(gym.Env):
    
    def __init__(
        self,
        node_name: str = 'ForceGraspEnv',
        package_name: str = 'birmingham_cell_tests',
        distance_threshold: float = 0.005,
        epoch_len: int = None,
    ) -> None:
        rospy.init_node(node_name)

        self.in_grasp_area = False
                
        rospack = rospkg.RosPack()
        self.package_path = rospack.get_path(package_name)
        # /home/gauss/projects/personal_ws/src/birmingham_cell/birmingham_cell_tests
        self.distance_threshold = distance_threshold
        self.step_number = 0
        self.epoch_len = epoch_len
        self.epoch_steps = 0
               
        self.param_upper_bound = np.array([ 0.05, 0.05, 0.05, 0.05])
        self.param_lower_bound = np.array([-0.05,-0.05,-0.05,-0.05])
        self.init_par_val      = np.array([ 0.00, 0.00, 0.00, 0.00])

        # lettura dati di forze
        data_path = self.package_path + '/data'
        peg_in_hole_data_path = data_path + '/1_peg_in_hole_data_02.csv'
        peg_in_hole_df = pd.read_csv(peg_in_hole_data_path)
        peg_in_hole_grouped = peg_in_hole_df.groupby('operation')
        grasp_df = peg_in_hole_grouped.get_group('grasping')
        insert_df = peg_in_hole_grouped.get_group('insertion')

        sampling_freq = 250
        grasp_max_time = 1
        grasp_df['gx'], grasp_df['gy'], grasp_df['gz'], grasp_df['ix'], grasp_df['iy'], grasp_df['iz'] = zip(*grasp_df['grasp_pose'].apply(self._extract_xyz))
        grasp_grouped = grasp_df.groupby('grasp_pose')  

        self.grasp_pose_to_forces = {}
        for name, group in grasp_grouped:
            xy_name = str(group.iloc[0]['gx']) + ',' + str(group.iloc[0]['gy']) + ',' + str(group.iloc[0]['ix']) + ',' + str(group.iloc[0]['iy'])
            self.grasp_pose_to_forces[xy_name] = {}
            self.grasp_pose_to_forces[xy_name]['fx'] = self._pad_forces(group['fx'].values, grasp_max_time * sampling_freq)
            self.grasp_pose_to_forces[xy_name]['fy'] = self._pad_forces(group['fy'].values, grasp_max_time * sampling_freq)
            self.grasp_pose_to_forces[xy_name]['fz'] = self._pad_forces(group['fz'].values, grasp_max_time * sampling_freq)
            self.grasp_pose_to_forces[xy_name]['tx'] = self._pad_forces(group['tx'].values, grasp_max_time * sampling_freq)
            self.grasp_pose_to_forces[xy_name]['ty'] = self._pad_forces(group['ty'].values, grasp_max_time * sampling_freq)
            self.grasp_pose_to_forces[xy_name]['tz'] = self._pad_forces(group['tz'].values, grasp_max_time * sampling_freq)

        insert_max_time = 2
        insert_df['gx'], insert_df['gy'], insert_df['gz'], insert_df['ix'], insert_df['iy'], insert_df['iz'] = zip(*insert_df['grasp_pose'].apply(self._extract_xyz))
        insert_grouped = insert_df.groupby('grasp_pose')  

        self.insert_pose_to_forces = {}
        for name, group in insert_grouped:
            xy_name = str(group.iloc[0]['gx']) + ',' + str(group.iloc[0]['gy']) + ',' + str(group.iloc[0]['ix']) + ',' + str(group.iloc[0]['iy'])
            self.insert_pose_to_forces[xy_name] = {}
            self.insert_pose_to_forces[xy_name]['fx'] = self._pad_forces(group['fx'].values, insert_max_time * sampling_freq)
            self.insert_pose_to_forces[xy_name]['fy'] = self._pad_forces(group['fy'].values, insert_max_time * sampling_freq)
            self.insert_pose_to_forces[xy_name]['fz'] = self._pad_forces(group['fz'].values, insert_max_time * sampling_freq)
            self.insert_pose_to_forces[xy_name]['tx'] = self._pad_forces(group['tx'].values, insert_max_time * sampling_freq)
            self.insert_pose_to_forces[xy_name]['ty'] = self._pad_forces(group['ty'].values, insert_max_time * sampling_freq)
            self.insert_pose_to_forces[xy_name]['tz'] = self._pad_forces(group['tz'].values, insert_max_time * sampling_freq)

        self.theoretical_correct_grasp_pos  = np.array([0.0,0.0])
        self.theoretical_correct_insert_pos = np.array([0.0,0.0])
        
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
        grasp_pose_err_low_limit = [-0.019, -0.019]
        grasp_pose_err_high_limit = [0.019, 0.019]
        insert_pose_err_low_limit = [-0.019, -0.0019]
        insert_pose_err_high_limit = [0.019, 0.0019]

        obj_pos_error = np.ndarray.tolist(self.np_random.uniform(grasp_pose_err_low_limit, grasp_pose_err_high_limit))
        self.correct_grasp_pos  = self.theoretical_correct_grasp_pos  + obj_pos_error
        self.initial_grasp_pos  = copy.copy(self.theoretical_correct_grasp_pos)
        self.current_grasp_pos  = copy.copy(self.initial_grasp_pos)

        tar_pos_error = np.ndarray.tolist(self.np_random.uniform(insert_pose_err_low_limit, insert_pose_err_high_limit))
        self.correct_insert_pos  = self.theoretical_correct_insert_pos  + tar_pos_error
        self.initial_insert_pos  = copy.copy(self.theoretical_correct_insert_pos)
        self.current_insert_pos  = copy.copy(self.initial_insert_pos)

        self.param_values = copy.copy(self.init_par_val)

        current_to_correct_grasp_distance = self.current_grasp_pos - self.correct_grasp_pos
        current_to_correct_insert_distance = self.current_insert_pos - self.correct_insert_pos

        calculed_grasp_forces, self.in_grasp_area, _, _  = self._generate_forces(current_to_correct_grasp_distance[0],
                                                                          current_to_correct_grasp_distance[1],
                                                                          current_to_correct_insert_distance[0],
                                                                          current_to_correct_insert_distance[1],
                                                                          self.grasp_pose_to_forces)
        self.grasp_forces = np.concatenate([calculed_grasp_forces['fx'],
                                            calculed_grasp_forces['fy'],
                                            calculed_grasp_forces['fz'],
                                            calculed_grasp_forces['tx'],
                                            calculed_grasp_forces['ty'],
                                            calculed_grasp_forces['tz'],])
        _, _, calculed_insert_forces, self.in_insert_area = self._generate_forces(current_to_correct_grasp_distance[0],
                                                                            current_to_correct_grasp_distance[1],
                                                                            current_to_correct_insert_distance[0],
                                                                            current_to_correct_insert_distance[1],
                                                                            self.insert_pose_to_forces)
        self.insert_forces = np.concatenate([calculed_insert_forces['fx'],
                                             calculed_insert_forces['fy'],
                                             calculed_insert_forces['fz'],
                                             calculed_insert_forces['tx'],
                                             calculed_insert_forces['ty'],
                                             calculed_insert_forces['tz'],])

        observation = self._get_obs()
        info = {"is_success": False}
        return observation, info
    
    def _is_success(self) -> np.array:
        if (self._distance(self.current_grasp_pos,self.correct_grasp_pos) < self.distance_threshold) and (self._distance(self.current_insert_pos,self.correct_insert_pos) < self.distance_threshold):
            success = True
        else:
            success = False
        return np.array(success, dtype=bool)

    def step(self, action: np.array) -> Tuple[Dict[str, np.array], float, bool, bool, Dict[str, Any]]:
        self.epoch_steps += 1

        self.param_values = np.add(self.param_values, np.multiply(action, self.max_variations))
        self.param_values = np.clip(self.param_values, self.param_lower_bound, self.param_upper_bound)
        self.current_grasp_pos[0] = self.initial_grasp_pos[0] + self.param_values[0]
        self.current_grasp_pos[1] = self.initial_grasp_pos[1] - self.param_values[1]
        self.current_insert_pos[0] = self.initial_insert_pos[0] + self.param_values[2]
        self.current_insert_pos[1] = self.initial_insert_pos[1] - self.param_values[3]
        current_to_correct_grasp_distance = self.current_grasp_pos - self.correct_grasp_pos
        current_to_correct_insert_distance = self.current_insert_pos - self.correct_insert_pos
        calculed_grasp_forces, self.in_grasp_area, _, _  = self._generate_forces(current_to_correct_grasp_distance[0],
                                                                          current_to_correct_grasp_distance[1],
                                                                          current_to_correct_insert_distance[0],
                                                                          current_to_correct_insert_distance[1],
                                                                          self.grasp_pose_to_forces)
        self.grasp_forces = np.concatenate([calculed_grasp_forces['fx'],
                                            calculed_grasp_forces['fy'],
                                            calculed_grasp_forces['fz'],
                                            calculed_grasp_forces['tx'],
                                            calculed_grasp_forces['ty'],
                                            calculed_grasp_forces['tz'],])
        _, _, calculed_insert_forces, self.in_insert_area = self._generate_forces(current_to_correct_grasp_distance[0],
                                                                            current_to_correct_grasp_distance[1],
                                                                            current_to_correct_insert_distance[0],
                                                                            current_to_correct_insert_distance[1],
                                                                            self.insert_pose_to_forces)
        self.insert_forces = np.concatenate([calculed_insert_forces['fx'],
                                             calculed_insert_forces['fy'],
                                             calculed_insert_forces['fz'],
                                             calculed_insert_forces['tx'],
                                             calculed_insert_forces['ty'],
                                             calculed_insert_forces['tz'],])

        observation = self._get_obs()

        success = bool(self._is_success())

        if ((self.epoch_len is not None) and success):
            single_reward = self._get_reward()
            remain_step = self.epoch_len - self.epoch_steps
            reward = single_reward + remain_step
        else:
            reward = self._get_reward()
        
        # print(self.grasp_forces)
        # print('current_grasp' + str(self.current_grasp_pos))
        # print('correct_grasp' + str(self.correct_grasp_pos))

        terminated = success
        truncated = False
        info = {"is_success": terminated}

        return observation, reward, terminated, truncated, info

    def _distance(self, pos1: np.array, pos2: np.array) -> float:
        return np.linalg.norm(pos1-pos2)
    
    def _get_reward(self) -> float:
        # if self.in_grasp_area and(self._distance(self.current_grasp_pos,self.correct_grasp_pos) < self.distance_threshold):
        #     reward = 1 - self._distance(self.current_insert_pos, self.correct_insert_pos)
        # elif self.in_grasp_area:
        #     reward = 0.5 - (self._distance(self.current_grasp_pos, self.correct_grasp_pos) * 5)
        #     reward -= (self._distance(self.current_insert_pos, self.theoretical_correct_insert_pos) * 0.5)
        if self.in_grasp_area and self.in_insert_area:
            reward = 1 - (self._distance(np.concatenate([self.current_grasp_pos,self.current_insert_pos]),
                                         np.concatenate([self.correct_grasp_pos,self.correct_insert_pos])))
            # reward = 1 - (self._distance(self.current_grasp_pos, self.correct_grasp_pos) * 2.5)
            # reward -= self._distance(self.current_insert_pos, self.correct_insert_pos) * 2.5
        else:
            reward = 0 - (self._distance(np.concatenate([self.current_grasp_pos,self.current_insert_pos]),
                                         np.concatenate([self.theoretical_correct_grasp_pos,self.theoretical_correct_insert_pos])))
            # reward = -0.5 - (self._distance(self.current_grasp_pos, self.theoretical_correct_grasp_pos) * 2.5)
            # reward -= (self._distance(self.current_insert_pos, self.theoretical_correct_insert_pos) * 2.5)
        return reward

    def _extract_xyz(self, position):
        pos = ast.literal_eval(position)
        return round(pos[0],3), round(pos[1],3), round(pos[2],3), round(pos[3],3), round(pos[4],3), round(pos[5],3),
  
    def _pad_forces(self, vector, lenght):
        if len(vector) > lenght:
            vector = vector[:lenght]
        return np.array(np.ndarray.tolist(vector) + [0.0] * (lenght - len(vector)))

    def _generate_forces(self, gx, gy, ix, iy, pose_to_forces):
        # print('x: ' + str(x))
        # print('y: ' + str(y))
        new_gx = round(gx / 0.005) * 0.005
        new_gy = round(gy / 0.005) * 0.005
        new_ix = round(ix / 0.005) * 0.005
        new_iy = round(iy / 0.005) * 0.005
        if new_gx < gx :
            gx1 = new_gx
            gx2 = new_gx + 0.005
        else:
            gx1 = new_gx - 0.005
            gx2 = new_gx
        if new_gy < gy :
            gy1 = new_gy
            gy2 = new_gy + 0.005
        else:
            gy1 = new_gy - 0.005
            gy2 = new_gy

        if new_ix < ix :
            ix1 = new_ix
            ix2 = new_ix + 0.005
        else:
            ix1 = new_ix - 0.005
            ix2 = new_ix
        if new_iy < iy :
            iy1 = new_iy
            iy2 = new_iy + 0.005
        else:
            iy1 = new_iy - 0.005
            iy2 = new_iy

        # for grasp
        u = (gx - gx1) / (gx2 - gx1)
        v = (gy - gy1) / (gy2 - gy1)

        name11 = str(gx1)+','+str(gy1)+','+str(new_ix)+','+str(new_iy)
        name12 = str(gx1)+','+str(gy2)+','+str(new_ix)+','+str(new_iy)
        name21 = str(gx2)+','+str(gy1)+','+str(new_ix)+','+str(new_iy)
        name22 = str(gx2)+','+str(gy2)+','+str(new_ix)+','+str(new_iy)

        if (name11 in pose_to_forces) and (name12 in pose_to_forces) and (name21 in pose_to_forces) and (name22 in pose_to_forces): 
            Fx11 = pose_to_forces[name11]['fx']
            Fx12 = pose_to_forces[name12]['fx']
            Fx21 = pose_to_forces[name21]['fx']
            Fx22 = pose_to_forces[name22]['fx']
            Fy11 = pose_to_forces[name11]['fy']
            Fy12 = pose_to_forces[name12]['fy']
            Fy21 = pose_to_forces[name21]['fy']
            Fy22 = pose_to_forces[name22]['fy']
            Fz11 = pose_to_forces[name11]['fz']
            Fz12 = pose_to_forces[name12]['fz']
            Fz21 = pose_to_forces[name21]['fz']
            Fz22 = pose_to_forces[name22]['fz']
            Tx11 = pose_to_forces[name11]['tx']
            Tx12 = pose_to_forces[name12]['tx']
            Tx21 = pose_to_forces[name21]['tx']
            Tx22 = pose_to_forces[name22]['tx']
            Ty11 = pose_to_forces[name11]['ty']
            Ty12 = pose_to_forces[name12]['ty']
            Ty21 = pose_to_forces[name21]['ty']
            Ty22 = pose_to_forces[name22]['ty']
            Tz11 = pose_to_forces[name11]['tz']
            Tz12 = pose_to_forces[name12]['tz']
            Tz21 = pose_to_forces[name21]['tz']
            Tz22 = pose_to_forces[name22]['tz']

            Fx1 = (1 - u) * Fx11 + u * Fx21
            Fx2 = (1 - u) * Fx12 + u * Fx22
            Fx  = (1 - v) * Fx1  + v * Fx2
            Fy1 = (1 - u) * Fy11 + u * Fy21
            Fy2 = (1 - u) * Fy12 + u * Fy22
            Fy  = (1 - v) * Fy1  + v * Fy2
            Fz1 = (1 - u) * Fz11 + u * Fz21
            Fz2 = (1 - u) * Fz12 + u * Fz22
            Fz  = (1 - v) * Fz1  + v * Fz2
            Tx1 = (1 - u) * Tx11 + u * Tx21
            Tx2 = (1 - u) * Tx12 + u * Tx22
            Tx  = (1 - v) * Tx1  + v * Tx2
            Ty1 = (1 - u) * Ty11 + u * Ty21
            Ty2 = (1 - u) * Ty12 + u * Ty22
            Ty  = (1 - v) * Ty1  + v * Ty2
            Tz1 = (1 - u) * Tz11 + u * Tz21
            Tz2 = (1 - u) * Tz12 + u * Tz22
            Tz  = (1 - v) * Tz1  + v * Tz2

            grasp_forces = {}
            grasp_forces['fx'] = Fx
            grasp_forces['fy'] = Fy
            grasp_forces['fz'] = Fz
            grasp_forces['tx'] = Tx
            grasp_forces['ty'] = Ty
            grasp_forces['tz'] = Tz

            in_grasp_dataset = True
        else:
            grasp_forces = {}
            grasp_forces['fx'] = np.zeros(len(pose_to_forces['0.0,0.0,0.0,0.0']['fx']))
            grasp_forces['fy'] = np.zeros(len(pose_to_forces['0.0,0.0,0.0,0.0']['fx']))
            grasp_forces['fz'] = np.zeros(len(pose_to_forces['0.0,0.0,0.0,0.0']['fx']))
            grasp_forces['tx'] = np.zeros(len(pose_to_forces['0.0,0.0,0.0,0.0']['fx']))
            grasp_forces['ty'] = np.zeros(len(pose_to_forces['0.0,0.0,0.0,0.0']['fx']))
            grasp_forces['tz'] = np.zeros(len(pose_to_forces['0.0,0.0,0.0,0.0']['fx']))
            in_grasp_dataset = False


        u = (ix - ix1) / (ix2 - ix1)
        v = (iy - iy1) / (iy2 - iy1)

        name11 = str(new_gx)+','+str(new_gy)+','+str(ix1)+','+str(iy1)
        name12 = str(new_gx)+','+str(new_gy)+','+str(ix1)+','+str(iy2)
        name21 = str(new_gx)+','+str(new_gy)+','+str(ix2)+','+str(iy1)
        name22 = str(new_gx)+','+str(new_gy)+','+str(ix2)+','+str(iy2)

        if (name11 in pose_to_forces) and (name12 in pose_to_forces) and (name21 in pose_to_forces) and (name22 in pose_to_forces): 
            Fx11 = pose_to_forces[name11]['fx']
            Fx12 = pose_to_forces[name12]['fx']
            Fx21 = pose_to_forces[name21]['fx']
            Fx22 = pose_to_forces[name22]['fx']
            Fy11 = pose_to_forces[name11]['fy']
            Fy12 = pose_to_forces[name12]['fy']
            Fy21 = pose_to_forces[name21]['fy']
            Fy22 = pose_to_forces[name22]['fy']
            Fz11 = pose_to_forces[name11]['fz']
            Fz12 = pose_to_forces[name12]['fz']
            Fz21 = pose_to_forces[name21]['fz']
            Fz22 = pose_to_forces[name22]['fz']
            Tx11 = pose_to_forces[name11]['tx']
            Tx12 = pose_to_forces[name12]['tx']
            Tx21 = pose_to_forces[name21]['tx']
            Tx22 = pose_to_forces[name22]['tx']
            Ty11 = pose_to_forces[name11]['ty']
            Ty12 = pose_to_forces[name12]['ty']
            Ty21 = pose_to_forces[name21]['ty']
            Ty22 = pose_to_forces[name22]['ty']
            Tz11 = pose_to_forces[name11]['tz']
            Tz12 = pose_to_forces[name12]['tz']
            Tz21 = pose_to_forces[name21]['tz']
            Tz22 = pose_to_forces[name22]['tz']

            Fx1 = (1 - u) * Fx11 + u * Fx21
            Fx2 = (1 - u) * Fx12 + u * Fx22
            Fx  = (1 - v) * Fx1  + v * Fx2
            Fy1 = (1 - u) * Fy11 + u * Fy21
            Fy2 = (1 - u) * Fy12 + u * Fy22
            Fy  = (1 - v) * Fy1  + v * Fy2
            Fz1 = (1 - u) * Fz11 + u * Fz21
            Fz2 = (1 - u) * Fz12 + u * Fz22
            Fz  = (1 - v) * Fz1  + v * Fz2
            Tx1 = (1 - u) * Tx11 + u * Tx21
            Tx2 = (1 - u) * Tx12 + u * Tx22
            Tx  = (1 - v) * Tx1  + v * Tx2
            Ty1 = (1 - u) * Ty11 + u * Ty21
            Ty2 = (1 - u) * Ty12 + u * Ty22
            Ty  = (1 - v) * Ty1  + v * Ty2
            Tz1 = (1 - u) * Tz11 + u * Tz21
            Tz2 = (1 - u) * Tz12 + u * Tz22
            Tz  = (1 - v) * Tz1  + v * Tz2

            insert_forces = {}
            insert_forces['fx'] = Fx
            insert_forces['fy'] = Fy
            insert_forces['fz'] = Fz
            insert_forces['tx'] = Tx
            insert_forces['ty'] = Ty
            insert_forces['tz'] = Tz

            in_insert_dataset = True
        else:
            insert_forces = {}
            insert_forces['fx'] = np.zeros(len(pose_to_forces['0.0,0.0,0.0,0.0']['fx']))
            insert_forces['fy'] = np.zeros(len(pose_to_forces['0.0,0.0,0.0,0.0']['fx']))
            insert_forces['fz'] = np.zeros(len(pose_to_forces['0.0,0.0,0.0,0.0']['fx']))
            insert_forces['tx'] = np.zeros(len(pose_to_forces['0.0,0.0,0.0,0.0']['fx']))
            insert_forces['ty'] = np.zeros(len(pose_to_forces['0.0,0.0,0.0,0.0']['fx']))
            insert_forces['tz'] = np.zeros(len(pose_to_forces['0.0,0.0,0.0,0.0']['fx']))
            in_insert_dataset = False

        return grasp_forces, in_grasp_dataset, insert_forces, in_insert_dataset

