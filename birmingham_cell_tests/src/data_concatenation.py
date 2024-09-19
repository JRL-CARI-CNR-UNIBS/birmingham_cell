#!/usr/bin/env python3

import pandas as pd

data1_name = '1_insert_data_xyz_eliminated'
data2_name = '3_insert_data_xyz'
data_concat_name = '1_3_insert_data_xyz'

pack_path = '/home/gauss/projects/personal_ws/src/birmingham_cell/birmingham_cell_tests'
data_path = pack_path +'/data/'

df1 = pd.read_csv(data_path + data1_name + '.csv')
df2 = pd.read_csv(data_path + data2_name + '.csv')

df_concat = pd.concat([df1, df2], axis=0)

df_concat.to_csv(data_path + data_concat_name + '.csv', index=False)