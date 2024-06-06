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
    return group.iloc[::6]

if __name__ == '__main__':

    generate_images = False
    print_max_times = False
    generate_vector_map = True
    
    data_path = '/home/gauss/projects/personal_ws/src/birmingham_cell/birmingham_cell_tests/data'
    grasp_data_path = data_path + '/00_grasp_data.csv'
    insert_data_path = data_path + '/00_insert_data.csv'

    grasp_df = pd.read_csv(grasp_data_path)
    grasp_grouped = grasp_df.groupby('grasp_pose')  
    reduced_grasp_groups = [reduce_group(group) for _, group in grasp_grouped]
    reduced_grasp_df = pd.concat(reduced_grasp_groups).reset_index(drop=True)
    reduced_grasp_df.to_csv(data_path + '/02_grasp_data.csv', index=False)

    insert_df = pd.read_csv(insert_data_path)
    insert_grouped = insert_df.groupby('insert_pose')  
    reduced_insert_groups = [reduce_group(group) for _, group in insert_grouped]
    reduced_insert_df = pd.concat(reduced_insert_groups).reset_index(drop=True)
    reduced_insert_df.to_csv(data_path + '/02_insert_data.csv', index=False)