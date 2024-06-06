#!/usr/bin/env python3

import numpy as np
import sys

import rospkg
import yaml

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
    global current_considered_pos_name
    global current_considered_pos_value
    info = {current_considered_pos_name: current_considered_pos_value,
            'secs' : data.header.stamp.secs,
            'nsecs' : data.header.stamp.nsecs,
            'fx' : data.wrench.force.x,
            'fy' : data.wrench.force.y,
            'fz' : data.wrench.force.z,
            'tx' : data.wrench.torque.x,
            'ty' : data.wrench.torque.y,
            'tz' : data.wrench.torque.z,
    }
    wrench_record.append(info)


recording = False
wrench_record = []
current_considered_pos_name = ''
current_considered_pos_value = []

if __name__ == '__main__':
    
    rospy.init_node('collect_grasping_data')

    params_path = sys.argv[1]

    rospack = rospkg.RosPack()
    pack_path = rospack.get_path('birmingham_cell_tests')
    file_path = pack_path + '/' + params_path

    with open(file_path) as file:
        params = yaml.safe_load(file)

    if 'target_model' in params:
        if 'name' in params['target_model']:
            tar_model_name = params['target_model']['name']
        else:
            tar_model_name = 'hole'
        if tar_model_name == 'cylinder_hole':
            if 'height' in params['target_model']:
                rospy.set_param('pybullet_simulation/objects/cylinder_hole/xacro_args/height', params['target_model']['height'])
            else:
                print('No height param for ' + tar_model_name + ' model')
                exit(1)
            if 'radius' in params['target_model']:
                rospy.set_param('pybullet_simulation/objects/cylinder_hole/xacro_args/radius', params['target_model']['radius'])
            else:
                print('No radius param for ' + tar_model_name + ' model')
                exit(1)
        elif tar_model_name == 'box_hole':
            if 'length' in params['target_model']:
                rospy.set_param('pybullet_simulation/objects/box_hole/xacro_args/length', params['target_model']['length'])
            else:
                print('No length param for ' + tar_model_name + ' model')
                exit(1)
            if 'width' in params['target_model']:
                rospy.set_param('pybullet_simulation/objects/box_hole/xacro_args/width', params['target_model']['width'])
            else:
                print('No width param for ' + tar_model_name + ' model')
                exit(1)
            if 'height' in params['target_model']:
                rospy.set_param('pybullet_simulation/objects/box_hole/xacro_args/height', params['target_model']['height'])
            else:
                print('No height param for ' + tar_model_name + ' model')
                exit(1)
    else:
        print('No target info')

    if 'object_model' in params:
        if 'name' in params['object_model']:
            obj_model_name = params['object_model']['name']
        else:
            obj_model_name = 'hole'
        if obj_model_name == 'cylinder':
            if 'height' in params['object_model']:
                rospy.set_param('pybullet_simulation/objects/cylinder/xacro_args/height', params['object_model']['height'])
            else:
                print('No height param for ' + obj_model_name + ' model')
                exit(1)
            if 'radius' in params['object_model']:
                rospy.set_param('pybullet_simulation/objects/cylinder/xacro_args/radius', params['object_model']['radius'])
            else:
                print('No radius param for ' + obj_model_name + ' model')
                exit(1)
            grasp_height = (params['object_model']['height'] / 2) - 0.015
        elif obj_model_name == 'box':
            if 'length' in params['object_model']:
                rospy.set_param('pybullet_simulation/objects/box/xacro_args/length', params['object_model']['length'])
            else:
                print('No length param for ' + obj_model_name + ' model')
                exit(1)
            if 'width' in params['object_model']:
                rospy.set_param('pybullet_simulation/objects/box/xacro_args/width', params['object_model']['width'])
            else:
                print('No width param for ' + obj_model_name + ' model')
                exit(1)
            if 'height' in params['object_model']:
                rospy.set_param('pybullet_simulation/objects/box/xacro_args/height', params['object_model']['height'])
            else:
                print('No height param for ' + obj_model_name + ' model')
                exit(1)
            grasp_height = (params['object_model']['height'] / 2) - 0.015
    else:
        print('No target info')

    new_tfs = []
    current_tfs = rospy.get_param('tf_params')
    for tf in current_tfs:
        new_tf = copy.copy(tf)
        if new_tf['name'] == 'can_grasp':
            new_tf['position'] = [0.0,0.0,grasp_height]
        new_tfs.append(new_tf)
    rospy.set_param('tf_params',new_tfs)
    correct_grasp_pos = np.array([0.0,0.0,grasp_height])
    correct_insertion_pos = np.array([0.00, 0.00, 0.17])
    
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
    object_names.append('hole')
    object_names.append('can')
    delete_model_clnt.call(object_names)

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
    spawn_model_clnt.call(object_names, model_name, pose, fixed)

    trees_path = pack_path + '/config/trees/collecting_data'
    run_tree_clnt.call('grasping_init', [trees_path])
    
    delete_state_clnt.call(['grasping_reset'])
    save_state_clnt.call('grasping_reset')

    grasping_area_lower_limit = [-0.02,-0.02]
    grasping_area_upper_limit = [ 0.02, 0.02]

    x_values = np.arange(grasping_area_lower_limit[0], grasping_area_upper_limit[0], 0.001)
    y_values = np.arange(grasping_area_lower_limit[1], grasping_area_upper_limit[1], 0.001)
    z_values = [0]
    combinations = list(itertools.product(x_values, y_values, z_values))
    print('combinations size: ' + str(len(combinations)))

    recording = False
    sub = rospy.Subscriber('/panda/panda_hand_joint/wrench',WrenchStamped, read_wrench_cb)

    current_considered_pos_name = 'grasp_pose'
    grasp_exec = 0
    print('grasp_exec:')
    # for grasp_pose_error in combinations: 
    #     grasp_exec += 1
    #     print('          ' + str(grasp_exec))
    #     current_considered_pos_value = np.ndarray.tolist(correct_grasp_pos + np.array(grasp_pose_error))
    #     current_tfs = rospy.get_param('tf_params')
    #     new_tfs = []
    #     for tf in current_tfs:
    #         new_tf = copy.copy(tf)
    #         if new_tf['name'] == 'can_grasp':
    #             new_tf['position'] = current_considered_pos_value
    #         new_tfs.append(new_tf)
    #     rospy.set_param('tf_params',new_tfs)

    #     result = run_tree_clnt.call('to_grasping', [trees_path])

    #     if result.result < 3:
    #         wrench_record = []
    #         recording = True

    #         run_tree_clnt.call('grasping', [trees_path])
            
    #         recording = False
    #         data = wrench_record    

    #         with open(pack_path + '/data/grasping_data' + str(grasp_exec) + '.csv', 'w') as csvfile:
    #             field_names = data[0].keys() if data else []

    #             csv_writer = csv.DictWriter(csvfile, fieldnames=field_names)
    
    #             if data:
    #                 csv_writer.writeheader()
                
    #             csv_writer.writerows(data)

    #     run_tree_clnt.call('out_of_grasping', [trees_path])

    #     restore_state_clnt.call('grasping_reset')

    new_tfs = []
    for tf in current_tfs:
        new_tf = copy.copy(tf)
        if new_tf['name'] == 'can_grasp':
            new_tf['position'] = np.ndarray.tolist(correct_grasp_pos)
        new_tfs.append(new_tf)
    rospy.set_param('tf_params',new_tfs)

    run_tree_clnt.call('insertion_init', [trees_path])

    delete_state_clnt.call(['insertion_reset'])
    save_state_clnt.call('insertion_reset')

    current_considered_pos_name = 'insert_pose'
    insert_exec = 0
    print('insert_exec:')

    for insertion_pose_error in combinations:       
        insert_exec += 1
        print('            ' + str(insert_exec))

        current_considered_pos_value = np.ndarray.tolist(correct_insertion_pos + np.array(insertion_pose_error))
        current_tfs = rospy.get_param('tf_params')
        new_tfs = []
        for tf in current_tfs:
            new_tf = copy.copy(tf)
            if new_tf['name'] == 'hole_insertion':
                new_tf['position'] = current_considered_pos_value
            new_tfs.append(new_tf)
        rospy.set_param('tf_params',new_tfs)

        result = run_tree_clnt.call('to_insertion', [trees_path])

        if result.result < 3:
            wrench_record = []
            recording = True
            
            run_tree_clnt.call('insertion', [trees_path])
            
            recording = False
            data = wrench_record    

            # with open(pack_path + '/data/insertion_data' + str(insert_exec) + '.csv', 'w') as csvfile:
            #     field_names = data[0].keys() if data else []

            #     csv_writer = csv.DictWriter(csvfile, fieldnames=field_names)
    
            #     if data:
            #         csv_writer.writeheader()
                
            #     csv_writer.writerows(data)

        restore_state_clnt.call('insertion_reset')
        restore_state_clnt.call('insertion_reset')
