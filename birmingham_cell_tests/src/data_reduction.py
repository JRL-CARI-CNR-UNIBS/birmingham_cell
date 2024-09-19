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


def reduce_group(group):
    global reduction
    return group.iloc[::reduction]

if __name__ == '__main__':
    reduction = 6
    data_name = '1_3_insert_data_xyz'
    new_data_name = data_name + '_' + str(reduction)
    pose_name = 'insert_pose'
    
    data_path = '/home/gauss/projects/personal_ws/src/birmingham_cell/birmingham_cell_tests/data'
    old_data_path = data_path + '/'+data_name+'.csv'

    df = pd.read_csv(old_data_path)
    grouped = df.groupby(pose_name)  
    reduced_groups = [reduce_group(group) for _, group in grouped]
    reduced_df = pd.concat(reduced_groups).reset_index(drop=True)
    reduced_df.to_csv(data_path+'/'+new_data_name+'.csv', index=False)

   
