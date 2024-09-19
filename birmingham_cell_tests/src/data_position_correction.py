#!/usr/bin/env python3

import pandas as pd

data_name = '1_insert_data'
position_name = 'insert'
decimal = 3

pack_path = '/home/gauss/projects/personal_ws/src/birmingham_cell/birmingham_cell_tests'
data_path = pack_path +'/data/'

df = pd.read_csv(data_path + data_name + '.csv')

df[['x', 'y', 'z']] = df[position_name + '_pose'].str.strip('[]').str.split(',', expand=True).astype(float)

df['x'] = df['x'].round(decimal)
df['y'] = df['y'].round(decimal)
df['z'] = df['z'].round(decimal)

df['position'] = df.apply(lambda row: f"[{row['x']},{row['y']},{row['z']}]", axis=1)

df.to_csv(data_path + data_name + '_xyz.csv', index=False)