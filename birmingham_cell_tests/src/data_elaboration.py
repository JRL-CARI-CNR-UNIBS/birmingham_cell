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
import ast

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    rospy.init_node('data_elaboration')

    rospack = rospkg.RosPack()
    pack_path = rospack.get_path('birmingham_cell_tests')
    data_path = pack_path +'/data/test_data.csv'
    data_path = '/home/gauss/Documents/RL_data/grasping_data.csv'

    # with open(data_path, 'r') as csvfile:
    #     csv_reader = csv.DictReader(csvfile)
    #     data = [row for row in csv_reader]
    #     data_ = [col for col in csv_reader]
    # print(data_)
    # for item in data_: 
    #     # pose = ast.literal_eval(item['insert_pose'])
    #     # item['insert_pose'] = [
    #     #     round(pose[0],3),
    #     #     round(pose[1],3),
    #     #     round(pose[2],3),            
    #     # ]
    #     # item['insert_pose'] = pose
    #     print(item)

    df = pd.read_csv(data_path)

    # poses = df['insert_pose'].unique()
    grouped = df.groupby('grasp_pose')

    # plt.figure(figsize=(10, 6))

    num = 0
    for name, group in grouped:
        # plt.clf()
        print(len(group))
        print(group)
    #     initial_time = group.iloc[0]['secs'] + (group.iloc[0]['nsecs'] * 10**-9)
    #     times = []
    #     for i in range(len(group)):
    #         times.append(group.iloc[i]['secs'] + (group.iloc[i]['nsecs'] * 10**-9) - initial_time)
    #     plt.plot(times,group['fx'].values, label=f'{name}')

    #     plt.xlabel('Time')
    #     plt.ylabel('Fx')
    #     plt.title('Storico delle Forze per Posizione {name}')
    #     plt.legend()
    #     plt.savefig(pack_path + '/data/images/img' + str(num) + '.png')
    #     num += 1
    # plt.close()

    exit(0)

    for i in range(len(poses)):
        pose = ast.literal_eval(poses[i])
        poses[i] = str([round(pose[0],3),
                        round(pose[1],3),
                        round(pose[2],3),
                        ])
    print(poses)




