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


class GeneralEnv(gym.Env):
    
    def __init__(
        self,
        epoch_len: int = None,
        space_dimension: int = 6,
        history_len: int = 10,
        single_threshold: float = 0.01,
        use_reward: bool = False,
    ) -> None:
        self.epoch_len = epoch_len
        self.epoch_steps = 0

        self.history_len = history_len
        self.use_reward = use_reward
        
        self.initial_param_values = [0.0] * space_dimension
        self.param_lower_bound    = [-1] * space_dimension
        self.param_upper_bound    = [ 1] * space_dimension
        self.param_values = copy.copy(self.initial_param_values)
        self.correct_param_value = copy.copy(self.initial_param_values)
        self.previous_reward = None
        self.current_reward = 0

        self.new_evaluation = 0
        self.old_evaluation = None

        self.threshold_vec = [single_threshold] * space_dimension

        self.param_value_history = np.ndarray.tolist(np.zeros(self.history_len * len(self.param_values)))

        space_division = 10.0
        self.max_variations = (np.array(self.param_upper_bound)-np.array(self.param_lower_bound))/space_division
        self.action_space = spaces.Box(-1, 1, shape=(len(self.max_variations),), dtype=np.float64)

        observation, _ = self.reset()  # required for init; seed can be changed later
        rospy.loginfo("Reset done")
        observation_shape = observation.shape
        self.observation_space = spaces.Box(-1, 1, shape=observation_shape, dtype=np.float64)

 
    def _get_obs(self) -> Dict[str, np.array]:
        # observation = np.concatenate([np.array(self.param_values),np.array(self.param_value_history),np.array(self.reward_history)])
        if self.use_reward:
            observation = np.concatenate([np.array([self.current_reward]),np.array(self.param_value_history),np.array(self.reward_diff_history)])
        else:
            observation = np.concatenate([np.array(self.param_value_history),np.array(self.reward_diff_history)])
            # observation = np.concatenate([np.array(self.param_value_history),np.array(self.reward_history)])
        return observation

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[Dict[str, np.array], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.reward_variation_weight = self.np_random.uniform(0.1,1)
        self.new_evaluation = 0
        self.old_evaluation = None
        self.epoch_steps = 0
        self.previous_reward = None
        self.param_values = copy.copy(self.initial_param_values)
        self.param_value_history = np.ndarray.tolist(np.zeros(self.history_len * len(self.param_values)))
        self.reward_history = np.ndarray.tolist(np.zeros(self.history_len))
        self.reward_diff_history = np.ndarray.tolist(np.zeros(self.history_len))
        self.correct_param_value = np.ndarray.tolist(self.np_random.uniform(self.param_lower_bound, self.param_upper_bound))
        self.initial_distance = self._distance(self.correct_param_value,self.param_values)
        observation = self._get_obs()
        info = {"is_success": False}
        return observation, info
    
    def _is_success(self) -> np.array:

        threshold_lower_limit = np.array(self.correct_param_value) - np.array(self.threshold_vec)
        threshold_upper_limit = np.array(self.correct_param_value) + np.array(self.threshold_vec)

        if (np.all(np.array(self.param_values) > threshold_lower_limit) & np.all(np.array(self.param_values) < threshold_upper_limit)):
            success = True
        else:
            success = False
        return np.array(success, dtype=bool)

    def step(self, action: np.array) -> Tuple[Dict[str, np.array], float, bool, bool, Dict[str, Any]]:
        self.last_action = copy.copy(action)
        self.epoch_steps += 1
        self.param_values = np.add(self.param_values, np.multiply(action, self.max_variations))
        self.param_values = np.clip(self.param_values, self.param_lower_bound, self.param_upper_bound)
        
        self.param_value_history = self.param_value_history[len(self.param_values):]
        self.param_value_history = np.ndarray.tolist(np.concatenate([np.array(self.param_value_history),np.array(self.param_values)]))

        success = bool(self._is_success())
        if ((self.epoch_len is not None) and success):
            single_reward = self._get_reward()
            remain_step = self.epoch_len - self.epoch_steps
            reward = single_reward + remain_step
        else:
            reward = self._get_reward()

        self.current_reward = copy.copy(reward)
        self.reward_history = self.reward_history[1:]
        self.reward_history.append(reward)

        if self.previous_reward is None:
            reward_diff = 0
        else:
            reward_diff = self.current_reward - self.previous_reward
        self.previous_reward = copy.copy(self.current_reward)
        self.reward_diff_history = self.reward_diff_history[1:]
        self.reward_diff_history.append(reward_diff)

        observation = self._get_obs()
        terminated = success
        truncated = False
        info = {"is_success": terminated}


        return observation, reward, terminated, truncated, info

    def _distance(self, pos1: np.array, pos2: np.array) -> float:
        return np.linalg.norm(np.array(pos1)-np.array(pos2))
    
    def _get_reward(self) -> float:
        current_distance = self._distance(self.correct_param_value,self.param_values)
        
        reward = (self.initial_distance - current_distance) / self.initial_distance

        return reward


