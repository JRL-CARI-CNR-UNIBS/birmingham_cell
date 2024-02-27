#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from threading import Thread
from threading import Lock
from rosgraph_msgs.msg import Clock

import numpy as np
import gymnasium as gym
import panda_gym
import time
import tf

def robot_jt_listener_cb(data, action, action_lock):
    action_lock.acquire()
    robot_target = data.position[:7]
    action[1] = np.array(robot_target)
    action_lock.release()


def robot_jt_listener(action,action_lock):
    rospy.Subscriber("/panda/joint_targets", 
                     JointState, 
                     lambda msg:
                         robot_jt_listener_cb(msg,
                                              action,
                                              action_lock))

def gripper_jt_listener_cb(data, action, action_lock):
    action_lock.acquire()
    gripper_target = data.position[:1]
    action[2] = np.array(gripper_target)
    action_lock.release()


def gripper_jt_listener(action,action_lock):
    rospy.Subscriber("/panda_gripper/joint_targets", 
                     JointState, 
                     lambda msg:
                         gripper_jt_listener_cb(msg,
                                                action,
                                                action_lock))

def simulation(env,action,action_lock,robot_js_pub,robot_js_msg,gripper_js_pub,gripper_js_msg):
    time_pub = rospy.Publisher('/clock', Clock, queue_size=10)
    rate = 500 
    step_time = 1 / rate
    # simulation_time = []
    # simulation_time.append(step_time)
    # simulation_time_msg = rospy.Time(simulation_time[0])
    simulation_time = 0.0
    simulation_time_msg = rospy.Time(simulation_time)

    time_pub.publish(simulation_time_msg)
    br = tf.TransformBroadcaster()
    
    while not rospy.is_shutdown():
        action_lock.acquire()
        action[0] = np.concatenate([action[1],action[2]])
        output = env.step(action[0])
        action_lock.release()
        # simulation_time[0] += step_time
        # simulation_time_msg = rospy.Time(simulation_time[0])
        simulation_time += step_time
        simulation_time_msg = rospy.Time(simulation_time)
        robot_js_msg.header.stamp = simulation_time_msg
        robot_js_msg.position = output[0]['robot_joint_pos'][:7]
        robot_js_msg.velocity = output[0]['robot_joint_vel'][:7]
        robot_js_msg.effort   = output[0]['robot_joint_eff'][:7]
        gripper_js_msg.header.stamp = simulation_time_msg
        gripper_js_msg.position = [output[0]['robot_joint_pos'][7]]
        gripper_js_msg.velocity = [output[0]['robot_joint_vel'][7]]
        gripper_js_msg.effort   = [output[0]['robot_joint_eff'][7]]
        time_pub.publish(simulation_time_msg)
        robot_js_pub.publish(robot_js_msg)
        gripper_js_pub.publish(gripper_js_msg)
        # print(output[0]['achieved_goal'][0)
        # print(output[0]['achieved_goal'])
        # print(output[0])

        br.sendTransform((output[0]['achieved_goal'][:3]),
                         (output[0]['achieved_goal'][3,7]),
                         rospy.Time.now(),
                         'object',
                         "world") 
        br.sendTransform((output[0]['desired_goal']),
                         (0,0,0,1),
                         rospy.Time.now(),
                         'target',
                         "world") 


if __name__ == '__main__':
    rospy.init_node('pybullet_world')
    robot_js_pub = rospy.Publisher('/panda/joint_states', JointState, queue_size=1)
    gripper_js_pub = rospy.Publisher('/panda_gripper/joint_states', JointState, queue_size=1)

    env = gym.make('PandaPegInHoleJointsTarget-v3',render_mode='human')
    output = env.reset()
    robot_js_msg = JointState()
    robot_js_msg.header = Header()
    robot_js_msg.header.stamp = rospy.Time.now()
    robot_js_msg.name = ['panda_joint1',
                   'panda_joint2',
                   'panda_joint3',
                   'panda_joint4',
                   'panda_joint5',
                   'panda_joint6',
                   'panda_joint7',]
    gripper_js_msg = JointState()
    gripper_js_msg.header = Header()
    gripper_js_msg.header.stamp = rospy.Time.now()
    gripper_js_msg.name = ['panda_finger_joint1',]

    action = []
    action.append(output[0]['robot_joint_pos'][:7])
    action[0].append(0.0)
    action.append(output[0]['robot_joint_pos'][:7])
    action.append([0.0])
    action_lock = Lock()

    sim_thread = Thread(target=simulation, args=(env,
                                                 action,
                                                 action_lock,
                                                 robot_js_pub,
                                                 robot_js_msg,
                                                 gripper_js_pub,
                                                 gripper_js_msg,))
    sim_thread.start()

    robot_jt_listener(action, action_lock)
    gripper_jt_listener(action, action_lock)
    rospy.spin()

    sim_thread.join()































































