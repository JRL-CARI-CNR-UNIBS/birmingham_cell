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

def extract_xy(position):
    pos = ast.literal_eval(position)
    return round(pos[0],3), round(pos[1],3), round(pos[2],3)

def pad_forces(vector, lenght):
    return np.array(np.ndarray.tolist(vector) + [0.0] * (lenght - len(vector)))

if __name__ == '__main__':

    generate_images = True
    print_max_times = False
    generate_vector_map = False
    
    data_path = '/home/gauss/projects/personal_ws/src/birmingham_cell/birmingham_cell_tests/data'
    grasp_data_path = data_path + '/02_grasp_data.csv'
    insert_data_path = data_path + '/02_insert_data.csv'

    grasp_df = pd.read_csv(grasp_data_path)

    grasp_df['x'], grasp_df['y'], grasp_df['z'] = zip(*grasp_df['grasp_pose'].apply(extract_xy))
    grasp_unique_x = sorted(grasp_df['x'].unique())
    grasp_unique_y = sorted(grasp_df['y'].unique())
    grasp_x_mapping = {val: idx for idx, val in enumerate(grasp_unique_x)}
    grasp_y_mapping = {val: idx for idx, val in enumerate(grasp_unique_y)}

    grasp_grouped = grasp_df.groupby('grasp_pose')  

    insert_df = pd.read_csv(insert_data_path)

    insert_df['x'], insert_df['y'], insert_df['z'] = zip(*insert_df['insert_pose'].apply(extract_xy))
    insert_unique_x = sorted(insert_df['x'].unique())
    insert_unique_y = sorted(insert_df['y'].unique())
    insert_x_mapping = {val: idx for idx, val in enumerate(insert_unique_x)}
    insert_y_mapping = {val: idx for idx, val in enumerate(insert_unique_y)}

    insert_grouped = insert_df.groupby('insert_pose')  

    if print_max_times:
        grasping_times = []
        for name, group in grasp_grouped:
            initial_time = group.iloc[0]['secs'] + (group.iloc[0]['nsecs'] * 10**-9)
            times = []
            for i in range(len(group)):
                times.append(group.iloc[i]['secs'] + (group.iloc[i]['nsecs'] * 10**-9) - initial_time)
            grasping_times.append(times[-1])

        print('grasping_time :' + str(max(grasping_times)))

        insert_times = []
        for name, group in insert_grouped:
            initial_time = group.iloc[0]['secs'] + (group.iloc[0]['nsecs'] * 10**-9)
            times = []
            for i in range(len(group)):
                times.append(group.iloc[i]['secs'] + (group.iloc[i]['nsecs'] * 10**-9) - initial_time)
            insert_times.append(times[-1])

        print('insert_time :' + str(max(insert_times)))

    if generate_images:
        fig_fx, axes_fx = plt.subplots(nrows=len(grasp_unique_y), ncols=len(grasp_unique_x), figsize=(80, 80))
        fig_fy, axes_fy = plt.subplots(nrows=len(grasp_unique_y), ncols=len(grasp_unique_x), figsize=(80, 80))
        fig_fz, axes_fz = plt.subplots(nrows=len(grasp_unique_y), ncols=len(grasp_unique_x), figsize=(80, 80))
        fig_tx, axes_tx = plt.subplots(nrows=len(grasp_unique_y), ncols=len(grasp_unique_x), figsize=(80, 80))
        fig_ty, axes_ty = plt.subplots(nrows=len(grasp_unique_y), ncols=len(grasp_unique_x), figsize=(80, 80))
        fig_tz, axes_tz = plt.subplots(nrows=len(grasp_unique_y), ncols=len(grasp_unique_x), figsize=(80, 80))

        for name, group in grasp_grouped:
            ax_fx = axes_fx[grasp_y_mapping[group.iloc[0]['y']], grasp_x_mapping[group.iloc[0]['x']]]
            ax_fy = axes_fy[grasp_y_mapping[group.iloc[0]['y']], grasp_x_mapping[group.iloc[0]['x']]]
            ax_fz = axes_fz[grasp_y_mapping[group.iloc[0]['y']], grasp_x_mapping[group.iloc[0]['x']]]
            ax_tx = axes_tx[grasp_y_mapping[group.iloc[0]['y']], grasp_x_mapping[group.iloc[0]['x']]]
            ax_ty = axes_ty[grasp_y_mapping[group.iloc[0]['y']], grasp_x_mapping[group.iloc[0]['x']]]
            ax_tz = axes_tz[grasp_y_mapping[group.iloc[0]['y']], grasp_x_mapping[group.iloc[0]['x']]]
            initial_time = group.iloc[0]['secs'] + (group.iloc[0]['nsecs'] * 10**-9)
            times = []
            for i in range(len(group)):
                times.append(group.iloc[i]['secs'] + (group.iloc[i]['nsecs'] * 10**-9) - initial_time)
            # print('grasp_time: ' + str(times[-1]))
            ax_fx.plot(times,group['fx'].values, label=f'{name}')
            ax_fy.plot(times,group['fy'].values, label=f'{name}')
            ax_fz.plot(times,group['fz'].values, label=f'{name}')
            ax_tx.plot(times,group['tx'].values, label=f'{name}')
            ax_ty.plot(times,group['ty'].values, label=f'{name}')
            ax_tz.plot(times,group['tz'].values, label=f'{name}')

        fig_fx = ax_fx.get_figure()
        fig_fx.savefig(data_path + '/images/11fx_grasp.png')
        fig_fy = ax_fy.get_figure()
        fig_fy.savefig(data_path + '/images/11fy_grasp.png')
        fig_fz = ax_fz.get_figure()
        fig_fz.savefig(data_path + '/images/11fz_grasp.png')
        fig_tx = ax_tx.get_figure()
        fig_tx.savefig(data_path + '/images/11tx_grasp.png')
        fig_ty = ax_ty.get_figure()
        fig_ty.savefig(data_path + '/images/11ty_grasp.png')
        fig_tz = ax_tz.get_figure()
        fig_tz.savefig(data_path + '/images/11tz_grasp.png')

        # fig_fx.close()
        # fig_fy.close()
        # fig_fz.close()
        # fig_tx.close()
        # fig_ty.close()
        # fig_tz.close()

        fig_fx, axes_fx = plt.subplots(nrows=len(insert_unique_y), ncols=len(insert_unique_x), figsize=(80, 80))
        fig_fy, axes_fy = plt.subplots(nrows=len(insert_unique_y), ncols=len(insert_unique_x), figsize=(80, 80))
        fig_fz, axes_fz = plt.subplots(nrows=len(insert_unique_y), ncols=len(insert_unique_x), figsize=(80, 80))
        fig_tx, axes_tx = plt.subplots(nrows=len(insert_unique_y), ncols=len(insert_unique_x), figsize=(80, 80))
        fig_ty, axes_ty = plt.subplots(nrows=len(insert_unique_y), ncols=len(insert_unique_x), figsize=(80, 80))
        fig_tz, axes_tz = plt.subplots(nrows=len(insert_unique_y), ncols=len(insert_unique_x), figsize=(80, 80))

        for name, group in insert_grouped:
            ax_fx = axes_fx[insert_y_mapping[group.iloc[0]['y']], insert_x_mapping[group.iloc[0]['x']]]
            ax_fy = axes_fy[insert_y_mapping[group.iloc[0]['y']], insert_x_mapping[group.iloc[0]['x']]]
            ax_fz = axes_fz[insert_y_mapping[group.iloc[0]['y']], insert_x_mapping[group.iloc[0]['x']]]
            ax_tx = axes_tx[insert_y_mapping[group.iloc[0]['y']], insert_x_mapping[group.iloc[0]['x']]]
            ax_ty = axes_ty[insert_y_mapping[group.iloc[0]['y']], insert_x_mapping[group.iloc[0]['x']]]
            ax_tz = axes_tz[insert_y_mapping[group.iloc[0]['y']], insert_x_mapping[group.iloc[0]['x']]]
            initial_time = group.iloc[0]['secs'] + (group.iloc[0]['nsecs'] * 10**-9)
            times = []
            for i in range(len(group)):
                times.append(group.iloc[i]['secs'] + (group.iloc[i]['nsecs'] * 10**-9) - initial_time)
            # print('insert_time: ' + str(times[-1]))
            ax_fx.plot(times,group['fx'].values, label=f'{name}')
            ax_fy.plot(times,group['fy'].values, label=f'{name}')
            ax_fz.plot(times,group['fz'].values, label=f'{name}')
            ax_tx.plot(times,group['tx'].values, label=f'{name}')
            ax_ty.plot(times,group['ty'].values, label=f'{name}')
            ax_tz.plot(times,group['tz'].values, label=f'{name}')

        fig_fx = ax_fx.get_figure()
        fig_fx.savefig(data_path + '/images/12fx_insert.png')
        fig_fy = ax_fy.get_figure()
        fig_fy.savefig(data_path + '/images/12fy_insert.png')
        fig_fz = ax_fz.get_figure()
        fig_fz.savefig(data_path + '/images/12fz_insert.png')
        fig_tx = ax_tx.get_figure()
        fig_tx.savefig(data_path + '/images/12tx_insert.png')
        fig_ty = ax_ty.get_figure()
        fig_ty.savefig(data_path + '/images/12ty_insert.png')
        fig_tz = ax_tz.get_figure()
        fig_tz.savefig(data_path + '/images/12tz_insert.png')

    pose_to_forces = {}
    if generate_vector_map:
        for name, group in grasp_grouped:
            xy_name = str(group.iloc[0]['x']) + ',' + str(group.iloc[0]['y'])
            pose_to_forces[xy_name] = {}
            pose_to_forces[xy_name]['fx'] = pad_forces(group['fx'].values, 750)
            pose_to_forces[xy_name]['fy'] = pad_forces(group['fy'].values, 750)
            pose_to_forces[xy_name]['fz'] = pad_forces(group['fz'].values, 750)
            pose_to_forces[xy_name]['tx'] = pad_forces(group['tx'].values, 750)
            pose_to_forces[xy_name]['ty'] = pad_forces(group['ty'].values, 750)
            pose_to_forces[xy_name]['tz'] = pad_forces(group['tz'].values, 750)

        print(pose_to_forces)


