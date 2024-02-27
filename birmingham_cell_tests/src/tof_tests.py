#!/usr/bin/env python3

import rospy
import rospkg
import sys
from skills_util_msgs.srv import RunTree
from pybullet_simulation.srv import SpawnModel
from pybullet_simulation.srv import DeleteModel
from pybullet_simulation.srv import SaveState, DeleteState
from geometry_msgs.msg import Pose
import pyexcel_ods3 as od
import numpy as np

rospack = rospkg.RosPack()
path = rospack.get_path('birmingham_cell_tests')
path = path + '/src/python_classes'
sys.path.append(path)

import personal_class
import real_personal_class

def sample_target() -> np.array:
    """Sample a goal."""
    work_space_range_low  = np.array([0.3, -0.4, 0])
    work_space_range_high = np.array([0.6,  0.4, 0])
    goal = np.array([0.0, 0.0, 0.0])
    noise = np.random.uniform(work_space_range_low, work_space_range_high)
    goal += noise
    return goal

def sample_object(tar_pos) -> np.array:
    """Randomize start position of object."""
    work_space_range_low  = np.array([0.3, -0.4, 0])
    work_space_range_high = np.array([0.6,  0.4, 0])
    finish = False
    while not finish:
        object_position = np.array([0.0, 0.0, 0.0])
        noise = np.random.uniform(work_space_range_low, work_space_range_high)
        object_position += noise
        if (distance(tar_pos,object_position) > 0.25):
            finish = True
    return object_position

def distance(pos1: np.array, pos2: np.array) -> float:
    return np.linalg.norm(np.array(pos1)-np.array(pos2))


if __name__ == '__main__':

    tests_type = rospy.get_param('tests_type')

    rospy.init_node('optimizer_tests', anonymous=True)
    rospy.set_param('/optimization_end', False)

    spawn_model_clnt = rospy.ServiceProxy('/pybullet_spawn_model', SpawnModel)
    delete_model_clnt = rospy.ServiceProxy('/pybullet_delete_model', DeleteModel)

    start_exec_params = rospy.get_param('exec_params')

    run_tree_clnt = rospy.ServiceProxy('/skills_util/run_tree', RunTree)

    rospack = rospkg.RosPack()
    package_path = rospack.get_path('birmingham_cell_tests')
    tree_folder_path = package_path + '/config/trees'

    tree_name = rospy.get_param('/tree_name')

    run_tree_clnt.call('init_tree', [tree_folder_path])

    save_state_clnt = rospy.ServiceProxy('/pybullet_save_state', SaveState)
    delete_state_clnt = rospy.ServiceProxy('/pybullet_delete_state', DeleteState)

    test_names = ['test_1','test_2','test_3']

    iteration_number = 2

    for test_name in test_names:
        rospy.set_param('exec_params', start_exec_params)

        print('Test: ' + test_name)

        # Inserisco gli oggetti di scena in modo casuale
        target_name = 'hole'
        object_name = 'can'
        tar_pos = sample_target()
        obj_pos = sample_object(tar_pos)

        object_names = []
        object_names.append(target_name)  
        object_names.append(object_name)
        delete_model_clnt.call(object_names)

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
        model_name.append(target_name)
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
        model_name.append(object_name)
        pose.append(obj_pose)
        fixed.append(False)
        spawn_model_clnt.call(object_names, model_name, pose, fixed)
        print("Models spawned")

        # Devo inserire la randomizzazione delle posizioni di presa e rilascio 
        ########################################################################################################
        ########################################################################################################


        # salvo lo stato di start
        save_state_clnt.call('start')

        if (tests_type == 'simulation'):
            pers_c = personal_class.personal_optimizer(package_path, tree_name, test_name, iteration_number)
            if not pers_c.run_optimization():
                rospy.logerr('Failure during optimization process')
                exit(1)
        elif (tests_type == 'real'):
            pers_c = real_personal_class.real_personal_optimizer(package_path, tree_name, test_name, iteration_number)
            if not pers_c.run_optimization():
                rospy.logerr('Failure during optimization process')
                exit(1)

        # elimino lo stato di start visto che cambiamo le posizioni degli oggetti 
        delete_state_clnt.call(['start'])
        delete_model_clnt.call(object_names)

        ros_data = rospy.get_param(test_name)
        data = []
        data.append(ros_data[0].keys())
        for index in range(len(ros_data)):
            data.append(ros_data[index].values())
        if (tests_type == 'simulation'):
            data_ods = od.get_data(package_path + "/data/simulation_tests.ods")
            data_ods.update({test_name: data})
            print(package_path + "/data/simulation_tests.ods")
            print(type(data_ods))
            print(data_ods)
            od.save_data(package_path + "/data/simulation_tests.ods", data_ods)
        if (tests_type == 'real'):
            data_ods = od.get_data(package_path + "/data/real_tests.ods")
            data_ods.update({test_name: data})
            od.save_data(package_path + "/data/real_tests.ods", data_ods)

    rospy.set_param('/optimization_end', True)




