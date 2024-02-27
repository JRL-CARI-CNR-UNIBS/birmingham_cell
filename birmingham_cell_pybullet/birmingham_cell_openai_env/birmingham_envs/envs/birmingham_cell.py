#!/usr/bin/env python3
from typing import Any, Dict, Optional, Tuple

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from threading import Thread
from threading import Lock
from rosgraph_msgs.msg import Clock

import numpy as np
import gymnasium as gym
from gymnasium import spaces
# from gymnasium.utils import seeding

import time

class BirminghamCellEnv(gym.Env):
    
    def __init__(
        self,
    ) -> None:
        rospy.init_node("Birmingham_cell")
        self.js_pub = rospy.Publisher('/panda/joint_states', JointState, queue_size=1)
        self.time_pub = rospy.Publisher('/clock', Clock, queue_size=10)

        self.env = gym.make('PandaPegInHoleJointsTarget-v3',render_mode='human')

        # initial_state = env.reset()
        self.joint_targets = []
        self.joint_targets.append(self.env.reset()[0]['robot_config'][:8])
        self.js_msg = JointState()

        self.action_lock = Lock()
        self.sim_thread = Thread(target=self.simulation)
        self.sim_thread.start()
        self.env_state = tuple()

        observation, _ = self.reset()  # required for init; seed can be changed later
        observation_shape = observation["observation"].shape
        achieved_goal_shape = observation["achieved_goal"].shape
        desired_goal_shape = observation["desired_goal"].shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
                desired_goal=spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
                achieved_goal=spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
                robot_config=spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
            )
        )
        self.action_space = spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32)

    def simulation(self):
        in_time = time.time()
        rate = 500 
        step_time = 1 / rate
        simulation_time = []
        simulation_time.append(step_time)
        simulation_time_msg = rospy.Time(simulation_time[0])
        self.time_pub.publish(simulation_time_msg)
        self.js_msg.header = Header()
        self.js_msg.name = ['panda_joint1',
                            'panda_joint2',
                            'panda_joint3',
                            'panda_joint4',
                            'panda_joint5',
                            'panda_joint6',
                            'panda_joint7',
                            'panda_finger_joint1']

        while not rospy.is_shutdown():
            self.action_lock.acquire()
            self.env_state = self.env.step(self.joint_targets[0])
            self.action_lock.release()
            simulation_time[0] += step_time
            simulation_time_msg = rospy.Time(simulation_time[0])
            self.time_pub.publish(simulation_time_msg)
            self.js_msg.header.stamp = rospy.Time.now()
            self.js_msg.position = self.env_state[0]['robot_config'][:8]
            self.js_msg.header.stamp = rospy.Time.now()
            self.js_pub.publish(self.js_msg)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return self.env_state[0]

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        return self.env.reset()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        return self.env_state

    def render(self) -> Optional[np.ndarray]:
        """Render.

        If render mode is "rgb_array", return an RGB array of the scene. Else, do nothing and return None.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        return 
