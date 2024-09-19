#!/usr/bin/env python3

import rospkg
import rospy
import ast

import pandas as pd

if __name__ == '__main__':
    dimension = 1600
    data_name = 'insertion_data'
    compaction_name = '3_insertion_data'

    pack_path = '/home/gauss/projects/personal_ws/src/birmingham_cell/birmingham_cell_tests'
    data_path = pack_path +'/data/'

    data_vec = []
    for i in range(1600):
        path = data_path + data_name + str(i+1) + '.csv'
        try:
            data = pd.read_csv(path)
            data_vec.append(data)
        except:
            continue

    df = pd.concat(data_vec, ignore_index=True)
    df.to_csv(data_path + compaction_name + '.csv')

