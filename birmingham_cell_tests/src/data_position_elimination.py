#!/usr/bin/env python3

import pandas as pd

data_name = '1_insert_data_xyz'
x_min, x_max = -0.005, 0.005
y_min, y_max = -0.005, 0.005

pack_path = '/home/gauss/projects/personal_ws/src/birmingham_cell/birmingham_cell_tests'
data_path = pack_path +'/data/'

df = pd.read_csv(data_path + data_name + '.csv')

df_filtered = df[~((df['x'] >= x_min) & (df['x'] <= x_max) & (df['y'] >= y_min) & (df['y'] <= y_max))]

df_filtered.to_csv(data_path + data_name + '_eliminated.csv', index=False)