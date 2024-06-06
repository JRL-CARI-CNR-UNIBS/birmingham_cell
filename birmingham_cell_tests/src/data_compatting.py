#!/usr/bin/env python3

import rospkg
import rospy
import ast

import pandas as pd

if __name__ == '__main__':
    pack_path = '/home/gauss/projects/personal_ws/src/birmingham_cell/birmingham_cell_tests'
    data_path = pack_path +'/data/'

    grasping_data = []
    insertion_data = []
    for i in range(1600):
        grasp_path = data_path + 'grasping_data' + str(i+1) + '.csv'
        try:
            data = pd.read_csv(grasp_path)
            grasping_data.append(data)
        except:
            continue

    print(len(grasping_data))
    grasp_df = pd.concat(grasping_data, ignore_index=True)
    grasp_df.to_csv(data_path + '00_grasp_data.csv')

    for i in range(1600):
        insert_path = data_path + 'insertion_data' + str(i+1) + '.csv'

        try:
            data = pd.read_csv(insert_path)
            insertion_data.append(data)
        except:
            continue

    print(len(insertion_data))
    insert_df = pd.concat(insertion_data, ignore_index=True)
    insert_df.to_csv(data_path + '00_insert_data.csv')