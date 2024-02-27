#!/usr/bin/env python3

import os
import rospy
import rospkg
from skills_util_msgs.srv import RunTree
from pybullet_simulation.srv import RestoreState


class real_personal_optimizer:
    def __init__(self, package_path, task_name, tree_name, data_name, iteration):

        self.package_path = package_path
        self.tree_name    = tree_name
        self.data_name    = data_name
        self.iteration    = iteration
        self.task_name    = task_name

        self.run_tree_clnt = rospy.ServiceProxy('/skills_util/run_tree', RunTree)

        data = []
        self.data_path = package_path + '/' + self.task_name + '/data/'
        self.tree_folder_path = package_path + '/' + self.task_name + '/config/trees'

        if (self.data_name + '.yaml') in os.listdir(self.data_path):
            if (self.data_name + '_RL_params.yaml') in os.listdir(self.data_path):
                print('Load params')
                os.system('rosparam load ' + self.data_path + '/' + self.data_name + '.yaml /' + self.data_name)
                print('History params loaded')
                os.system('rosparam load ' + self.data_path + '/' + self.data_name + '_RL_params.yaml /RL_params')
                print('RL params loaded')
            else:
                print('No RL params file. It remain the start one.')
                rospy.set_param(self.data_name, data)
        else:
            print('No data params file')
            rospy.set_param(self.data_name, data)

        self.param_names = []
        self.data_param_names = []
        all_param_names = rospy.get_param_names()

        for param_name in all_param_names:
            if (param_name.find('actions') != -1):
                self.param_names.append(param_name)
                param_name = param_name.replace('RL_params/actions/', '')
                param_name = param_name.replace('skills/', '')
                self.data_param_names.append(param_name)


    def run_tree(self):
#        print('run_tree:')
#        print(self.tree_name)
#        print(self.tree_folder_path)
        result = self.run_tree_clnt.call(self.tree_name, [self.tree_folder_path])

        if not result:
            print('ValueError("Error tree")')
            return ValueError("Error tree")

        if (result.result == 3):
            rospy.logerr('Failure with the tree execution')
            return False

        all_param_names = rospy.get_param_names()
        reward_names = []
        rewards = []
        for param_name in all_param_names:
            if (param_name.find('total_reward') != -1):
                rewards.append(rospy.get_param(param_name))
                reward_names.append(param_name.replace('RL_params/actions/', ''))

        data = rospy.get_param(self.data_name)
        iteration = len(data) + 1

        current_data = {}
        for i in range(len(self.param_names)):
            param_value = rospy.get_param(self.param_names[i])
            if type(param_value) is float:
                current_data[self.data_param_names[i]] = param_value
            elif type(param_value) is int:
                current_data[self.data_param_names[i]] = param_value
            elif type(param_value) is list:
                for x in range(len(param_value)):
                    current_data[self.data_param_names[i] + '_' + str(x)] = param_value[x]
#            else:
#                print(param_name + ': ' + str(type(param_value)))

        current_data['iteration'] = iteration
        for i in range(len(reward_names)):
            current_data[reward_names[i]] = rewards[i]

        data.append(current_data)
        rospy.set_param(self.data_name, data)

        return True

    def run_optimization(self): 
        data = rospy.get_param(self.data_name)
        missing_iterations = self.iteration - len(data)
        print('missing_iterations:')
        print(missing_iterations)

        restore_state_clnt = rospy.ServiceProxy('/pybullet_restore_state', RestoreState)

        for i in range(missing_iterations):
            if not self.run_tree():
                return False
            os.system('rosparam dump ' + self.data_path + '/' + self.data_name + '.yaml /' + self.data_name)
            os.system('rosparam dump ' + self.data_path + '/' + self.data_name + '_RL_params.yaml /RL_params')
            restore_state_clnt.call('start')

        return True
