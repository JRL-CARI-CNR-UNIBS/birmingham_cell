#!/usr/bin/env python3

import numpy as np
import sys

import rospkg
import yaml

import os
import csv
import threading
import copy
import rospy
from skills_util_msgs.srv import RunTree
from pybullet_simulation.srv import SpawnModel, DeleteModel, SaveState, RestoreState, DeleteState
from geometry_msgs.msg import Pose, WrenchStamped
import itertools

def read_wrench_cb(data):
    global wrench_record
    global current_considered_pos_value
    global current_operation_name
    global recording
    info = {'pose': current_considered_pos_value,
            'operation': current_operation_name,
            'secs' : data.header.stamp.secs,
            'nsecs' : data.header.stamp.nsecs,
            'fx' : data.wrench.force.x,
            'fy' : data.wrench.force.y,
            'fz' : data.wrench.force.z,
            'tx' : data.wrench.torque.x,
            'ty' : data.wrench.torque.y,
            'tz' : data.wrench.torque.z,
    }
    if recording:
        wrench_record.append(info)

def set_tf(tf_name,tf_value):
    current_tfs = rospy.get_param('tf_params')
    new_tfs = []
    for tf in current_tfs:
        new_tf = copy.copy(tf)
        if new_tf['name'] == tf_name:
            new_tf['position'] = tf_value
        new_tfs.append(new_tf)
    rospy.set_param('tf_params',new_tfs)


def iterative_registration(default_pose,error_combinations,out_of_grasping):
    global recording
    global current_considered_pos_name
    global current_date_name
    global current_operation_name
    global current_considered_pos_value
    global wrench_record
    global run_tree_clnt
    global trees_path

    run_tree_clnt.call('to_' + current_operation_name + '_approach', [trees_path])

    delete_state_clnt.call(['reset_state'])
    save_state_clnt.call('reset_state')

    execution = 0
    print(current_operation_name + ': ')
    for pose_error in error_combinations: 
        execution += 1
        print('          ' + str(execution))
        current_considered_pos_value = np.ndarray.tolist(default_pose + np.array(pose_error))
        set_tf(current_considered_pos_name,current_considered_pos_value)

        result = run_tree_clnt.call('to_' + current_operation_name, [trees_path])

        if result.result < 3:
            os.makedirs(pack_path + '/data/' + current_date_name, exist_ok=True)
            wrench_record = []
            recording = True

            run_tree_clnt.call(current_operation_name, [trees_path])
            
            recording = False
            data = wrench_record    

            with open(pack_path + '/data/' + current_date_name + '/exec_' + str(execution) + '.csv', 'w') as csvfile:
                field_names = data[0].keys() if data else []

                csv_writer = csv.DictWriter(csvfile, fieldnames=field_names)
    
                if data:
                    csv_writer.writeheader()
                
                csv_writer.writerows(data)
        if out_of_grasping:
            run_tree_clnt.call('out_of_grasping', [trees_path])

        restore_state_clnt.call('reset_state')
        restore_state_clnt.call('reset_state')


recording = False
wrench_record = []
current_date_name = ''
current_considered_pos_name = ''
current_considered_pos_value = []
current_operation_name = ''
run_tree_clnt = None
trees_path = ''

if __name__ == '__main__':
    
    rospy.init_node('collect_data')

    rospack = rospkg.RosPack()
    pack_path = rospack.get_path('birmingham_cell_tests')
    trees_path = pack_path + '/config/trees/complex_task'

    container1_default_grasp_pose  = [0.00, 0.00, 0.04]
    container1_default_insert_pose = [0.00, 0.00, 0.10]
    item_default_grasp_pose        = [0.00, 0.00, 0.04]
    item_default_insert_pose       = [0.00, 0.00, 0.11]
    container2_default_grasp_pose  = [0.00, 0.00, 0.04]
    container2_default_insert_pose = [0.00, 0.00, 0.10]

    container1_grasping_area_lower_limit  = [-0.005,-0.005, 0]
    container1_grasping_area_upper_limit  = [ 0.005, 0.005, 0]
    container1_insertion_area_lower_limit = [-0.005,-0.005, 0]
    container1_insertion_area_upper_limit = [ 0.005, 0.005, 0]
    item_grasping_area_lower_limit        = [-0.005,-0.005, 0]
    item_grasping_area_upper_limit        = [ 0.005, 0.005, 0]
    item_insertion_area_lower_limit       = [-0.005,-0.005, 0]
    item_insertion_area_upper_limit       = [ 0.005, 0.005, 0]
    container2_grasping_area_lower_limit  = [-0.005,-0.005, 0]
    container2_grasping_area_upper_limit  = [ 0.005, 0.005, 0]
    container2_insertion_area_lower_limit = [-0.005,-0.005, 0]
    container2_insertion_area_upper_limit = [ 0.005, 0.005, 0]
    
    # container1_grasping_area_lapse  = 0.00025
    container1_grasping_area_lapse  = 0.005
    container1_insertion_area_lapse = 0.005
    item_grasping_area_lapse        = 0.005
    item_insertion_area_lapse       = 0.005
    container2_grasping_area_lapse  = 0.005
    container2_insertion_area_lapse = 0.005
    
    container1_grasp_x_values = np.arange(container1_grasping_area_lower_limit[0], container1_grasping_area_upper_limit[0], container1_grasping_area_lapse)
    container1_grasp_y_values = np.arange(container1_grasping_area_lower_limit[1], container1_grasping_area_upper_limit[1], container1_grasping_area_lapse)
    container1_grasp_z_values = [0]
    container1_error_grasp_combinations = list(itertools.product(container1_grasp_x_values, container1_grasp_y_values, container1_grasp_z_values))
    print('container1_grasp_combinations size: ' + str(len(container1_error_grasp_combinations)))
    item_grasp_x_values = np.arange(item_grasping_area_lower_limit[0], item_grasping_area_upper_limit[0], item_grasping_area_lapse)
    item_grasp_y_values = np.arange(item_grasping_area_lower_limit[1], item_grasping_area_upper_limit[1], item_grasping_area_lapse)
    item_grasp_z_values = [0]
    item_error_grasp_combinations = list(itertools.product(item_grasp_x_values, item_grasp_y_values, item_grasp_z_values))
    print('item_grasp_combinations size: ' + str(len(item_error_grasp_combinations)))
    container2_grasp_x_values = np.arange(container2_grasping_area_lower_limit[0], container2_grasping_area_upper_limit[0], container2_grasping_area_lapse)
    container2_grasp_y_values = np.arange(container2_grasping_area_lower_limit[1], container2_grasping_area_upper_limit[1], container2_grasping_area_lapse)
    container2_grasp_z_values = [0]
    container2_error_grasp_combinations = list(itertools.product(container2_grasp_x_values, container2_grasp_y_values, container2_grasp_z_values))
    print('container2_grasp_combinations size: ' + str(len(container2_error_grasp_combinations)))

    container1_insertion_x_values = np.arange(container1_insertion_area_lower_limit[0], container1_insertion_area_upper_limit[0], container1_insertion_area_lapse)
    container1_insertion_y_values = np.arange(container1_insertion_area_lower_limit[1], container1_insertion_area_upper_limit[1], container1_insertion_area_lapse)
    container1_insertion_z_values = [0]
    container1_error_insertion_combinations = list(itertools.product(container1_insertion_x_values, container1_insertion_y_values, container1_insertion_z_values))
    print('container1_insertion_combinations size: ' + str(len(container1_error_insertion_combinations)))
    item_insertion_x_values = np.arange(item_insertion_area_lower_limit[0], item_insertion_area_upper_limit[0], item_insertion_area_lapse)
    item_insertion_y_values = np.arange(item_insertion_area_lower_limit[1], item_insertion_area_upper_limit[1], item_insertion_area_lapse)
    item_insertion_z_values = [0]
    item_error_insertion_combinations = list(itertools.product(item_insertion_x_values, item_insertion_y_values, item_insertion_z_values))
    print('item_insertion_combinations size: ' + str(len(item_error_insertion_combinations)))
    container2_insertion_x_values = np.arange(container2_insertion_area_lower_limit[0], container2_insertion_area_upper_limit[0], container2_insertion_area_lapse)
    container2_insertion_y_values = np.arange(container2_insertion_area_lower_limit[1], container2_insertion_area_upper_limit[1], container2_insertion_area_lapse)
    container2_insertion_z_values = [0]
    container2_error_insertion_combinations = list(itertools.product(container2_insertion_x_values, container2_insertion_y_values, container2_insertion_z_values))
    print('container2_insertion_combinations size: ' + str(len(container2_error_insertion_combinations)))

    recording = False
    sub = rospy.Subscriber('/panda/panda_hand_joint/wrench',WrenchStamped, read_wrench_cb)

    rospy.loginfo("Wait for skills_util/run_tree service")
    rospy.wait_for_service('/skills_util/run_tree')
    run_tree_clnt = rospy.ServiceProxy('/skills_util/run_tree', RunTree)
    rospy.loginfo("Connected")
    rospy.loginfo("Wait for pybullet_spawn_model service")
    rospy.wait_for_service('/pybullet_spawn_model')
    spawn_model_clnt = rospy.ServiceProxy('/pybullet_spawn_model', SpawnModel)
    rospy.loginfo("Connected")
    rospy.loginfo("Wait for pybullet_delete_model service")
    rospy.wait_for_service('/pybullet_delete_model')
    delete_model_clnt = rospy.ServiceProxy('/pybullet_delete_model', DeleteModel)
    rospy.loginfo("Connected")
    rospy.loginfo("Wait for pybullet_save_state service")
    rospy.wait_for_service('/pybullet_save_state')
    save_state_clnt = rospy.ServiceProxy('/pybullet_save_state', SaveState)
    rospy.loginfo("Connected")
    rospy.loginfo("Wait for pybullet_restore_state service")
    rospy.wait_for_service('/pybullet_restore_state')
    restore_state_clnt = rospy.ServiceProxy('/pybullet_restore_state', RestoreState)
    rospy.loginfo("Connected")
    rospy.loginfo("Wait for pybullet_delete_state service")
    rospy.wait_for_service('/pybullet_delete_state')
    delete_state_clnt = rospy.ServiceProxy('/pybullet_delete_state', DeleteState)
    rospy.loginfo("Connected")

    object_names = []
    object_names.append('accomodation')
    object_names.append('container1')
    object_names.append('item')
    object_names.append('container2')
    delete_model_clnt.call(object_names)

    run_tree_clnt.call('init',[trees_path])

    # inserisco nell'ambiente l'alloggio del contenitore
    object_names = []
    object_names.append('accomodation')
    delete_model_clnt.call(object_names)
    model_names = []
    poses = []
    fixed = []
    pose = Pose()
    pose.position.x = 0.5
    pose.position.y = 0.3
    pose.position.z = 0
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 1.0
    model_names.append('accomodation')
    poses.append(pose)
    fixed.append(True)
    spawn_model_clnt.call(object_names, model_names, poses, fixed)

    # inserisco nell'ambiente la parte bassa del contenitore
    object_names = []
    object_names.append('container1')
    delete_model_clnt.call(object_names)
    model_names = []
    poses = []
    fixed = []
    pose = Pose()
    pose.position.x = 0.5
    pose.position.y = -0.4
    pose.position.z = 0
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 1.0
    model_names.append('container1')
    poses.append(pose)
    fixed.append(False)
    spawn_model_clnt.call(object_names, model_names, poses, fixed)

    delete_state_clnt('before_container1_training')
    save_state_clnt('before_container1_training')
    current_considered_pos_name = 'container1_grasp'
    current_operation_name = 'container1_grasp'
    current_date_name = 'container1_grasping_training'
    iterative_registration(container1_default_grasp_pose,container1_error_grasp_combinations,True)
    restore_state_clnt('before_container1_training')

    current_date_name = 'container1_grasping_validation'
    iterative_registration(container1_default_grasp_pose,container1_error_grasp_combinations,True)
    restore_state_clnt('before_container1_training')

    set_tf(current_considered_pos_name,container1_default_grasp_pose)
    run_tree_clnt.call('container1_grasp_to_insertion',[trees_path])

    current_considered_pos_name = 'container1_insertion'
    current_operation_name = 'container1_insertion'
    current_date_name = 'container1_insertion_training'
    iterative_registration(container1_default_insert_pose,container1_error_insertion_combinations,False)
    restore_state_clnt('before_container1_training')

    current_date_name = 'container1_insertion_validation'
    iterative_registration(container1_default_insert_pose,container1_error_insertion_combinations,False)
    restore_state_clnt('before_container1_training')

    exit(0)

    object_names = []
    object_names.append('container1')
    delete_model_clnt.call(object_names)
    model_names = []
    poses = []
    fixed = []
    pose = Pose()
    pose.position.x = 0.5
    pose.position.y = 0.3
    pose.position.z = 0.025
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 1.0
    model_names.append('container1')
    poses.append(pose)
    fixed.append(False)
    spawn_model_clnt.call(object_names, model_names, poses, fixed)

    # inserisco nell'ambiente l'oggetto centrale
    object_names = []
    object_names.append('item')
    delete_model_clnt.call(object_names)
    model_names = []
    poses = []
    fixed = []
    pose = Pose()
    pose.position.x = 0.4
    pose.position.y = -0.4
    pose.position.z = 0
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 1.0
    model_names.append('item')
    poses.append(pose)
    fixed.append(False)
    spawn_model_clnt.call(object_names, model_names, poses, fixed)

    delete_state_clnt('before_item_training')
    save_state_clnt('before_item_training')
    # current_considered_pos_name = 'item_grasp'
    # current_operation_name = 'item_grasp'
    # current_date_name = 'item_grasping_training'
    # iterative_registration(item_default_grasp_pose,item_error_grasp_combinations,True)
    restore_state_clnt('before_item_training')

    # set_tf(current_considered_pos_name,item_default_grasp_pose)
    # run_tree_clnt.call('item_grasp_to_insertion',[trees_path])


    # current_considered_pos_name = 'item_insertion'
    # current_operation_name = 'item_insertion'
    # current_date_name = 'item_insertion_training'
    # iterative_registration(item_default_insert_pose,item_error_insertion_combinations,False)
    restore_state_clnt('before_item_training')


    object_names = []
    object_names.append('item')
    delete_model_clnt.call(object_names)
    model_names = []
    poses = []
    fixed = []
    pose = Pose()
    pose.position.x = 0.5
    pose.position.y = 0.3
    pose.position.z = 0.05
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 1.0
    model_names.append('item')
    poses.append(pose)
    fixed.append(False)
    spawn_model_clnt.call(object_names, model_names, poses, fixed)

    # inserisco nell'ambiente la parte alta del contenitore
    object_names = []
    object_names.append('container2')
    delete_model_clnt.call(object_names)
    model_names = []
    poses = []
    fixed = []
    pose = Pose()
    pose.position.x = 0.3
    pose.position.y = -0.4
    pose.position.z = 0
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 1.0
    model_names.append('container2')
    poses.append(pose)
    fixed.append(False)
    spawn_model_clnt.call(object_names, model_names, poses, fixed)


    delete_state_clnt('before_container2_training')
    save_state_clnt('before_container2_training')
    # current_considered_pos_name = 'container2_grasp'
    # current_operation_name = 'container2_grasp'
    # current_date_name = 'container2_grasping_training'
    # iterative_registration(container2_default_grasp_pose,container2_error_grasp_combinations,True)
    restore_state_clnt('before_container2_training')

    # set_tf(current_considered_pos_name,container2_default_grasp_pose)
    # run_tree_clnt.call('container2_grasp_to_insertion',[trees_path])

    # current_considered_pos_name = 'container2_insertion'
    # current_operation_name = 'container2_insertion'
    # current_date_name = 'container2_insertion_training'
    # iterative_registration(container2_default_insert_pose,container2_error_insertion_combinations,False)
    restore_state_clnt('before_container2_training')

    object_names = []
    object_names.append('container2')
    delete_model_clnt.call(object_names)
    model_names = []
    poses = []
    fixed = []
    pose = Pose()
    pose.position.x = 0.5
    pose.position.y = 0.3
    pose.position.z = 0.07
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 1.0
    model_names.append('container2')
    poses.append(pose)
    fixed.append(False)
    spawn_model_clnt.call(object_names, model_names, poses, fixed)

