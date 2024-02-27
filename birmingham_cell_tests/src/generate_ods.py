#!/usr/bin/env python3


import rospy 
import pyexcel_ods3 as od
import rospkg

rospy.init_node('up')
data = []
ros_data = rospy.get_param('test_1')
data.append(ros_data[0].keys())
for index in range(len(ros_data)):
    data.append(ros_data[index].values())

rospack = rospkg.RosPack()
package_path = rospack.get_path('battery_cell_tests')
tree_folder_path = package_path + '/can_pick_and_place/config/trees'
data_ods = od.get_data(package_path + '/can_pick_and_place/data/simulation_tests.ods')
data_ods.update({'test_1':data})
od.save_data(package_path + '/can_pick_and_place/data/simulation_tests.ods', data_ods)

