## 1. Importing the LonelyBoy Library (github.com/insert-generic-name-here/lonelyboy)
##    and all other Essential Libraries
import os, sys
sys.path.append(os.path.join(os.path.expanduser('~')))

from lonelyboy.geospatial import plots as gsplt
from lonelyboy.geospatial import preprocessing as gspp
from lonelyboy.timeseries import lbtimeseries as tspp
from lonelyboy.geospatial import group_patterns_v2 as gsgp

import psycopg2
import psycopg2.extras
import numpy as np
import configparser
import pandas as pd
import geopandas as gpd
import contextily as ctx
from random import choice
from tqdm import tqdm
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, shape
from haversine import haversine
from datetime import datetime, timedelta

from multiprocessing import cpu_count, Pool
from functools import partial
import datetime


### 1.5. Set the CLUSTER_ID
CLUSTER_ID = int(sys.argv[1]) # 0 = MASTER; 1-5 = SLAVES


### 2. Importing the Server Credentials & Connectiing to Server
properties = configparser.ConfigParser()
properties.read(os.path.join('..','sql_server.ini'))
properties = properties['SERVER']

host    = properties['host']
db_name = properties['db_name']
uname   = properties['uname']
pw      = properties['pw']
port    = properties['port']
con     = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port=port)
cur     = con.cursor()

### 2.5. Read MMSI list for each corresponding cluster and set up the MMSI Window Size
mmsi_list = pd.read_pickle(os.path.join('.', 'mmsi_list.pckl'))[CLUSTER_ID]
mmsi_list_window_size = 10


### 2.8. Define the essential queries in order to fetch or upload data
traj_sql = 'SELECT * FROM ais_data.dynamic_ships WHERE mmsi IN %s'
query_dynamic_ships_cleaned = "INSERT INTO ais_data.dynamic_ships_cleaned (mmsi, turn, speed, course, heading, lon, lat, ts, velocity) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"

for i in tqdm(range(0, len(mmsi_list), mmsi_list_window_size)):
    # print (f'Preprocessing: {i+mmsi_list_window_size}/{len(mmsi_list)} Vessels...')
    mmsis = mmsi_list[i:i+mmsi_list_window_size]
    # print (f'Vessel MMSIs: {mmsis}')

    traj = gpd.GeoDataFrame.from_postgis(traj_sql%(tuple(mmsis),), con, geom_col='geom')
    print ('Stage 1: Trajectory Preprocessing')

    # usual stuff
    traj.drop_duplicates(['mmsi','ts'], inplace=True)
    traj.sort_values('ts', inplace=True)
    traj.reset_index(inplace=True, drop=True)
    traj.drop(['id', 'status'], axis=1, inplace=True, errors='ignore')

    # calc 1st velocity, drop wrong entries
    traj = traj.groupby('mmsi', group_keys=False).apply(gspp.calculate_velocity)
    traj = traj.loc[traj.velocity<=50.0]
    
    # calc correct velocities (some points were droped)
    traj = traj.groupby('mmsi', group_keys=False).apply(gspp.calculate_velocity)
    
    print ('Checkpoint 1: Saving Cleansed Trajectories...')
    if os.path.exists(f'./test_data/nari_dynamic_clean_{CLUSTER_ID}.csv'):
        with open(f'./test_data/nari_dynamic_clean_{CLUSTER_ID}.csv', 'a') as f:
            traj.to_csv(f, header=False, index=False)
    else:
        print ('...creating file...\n')
        with open(f'./test_data/nari_dynamic_clean_{CLUSTER_ID}.csv', 'w') as f:
            traj.to_csv(f, header=True, index=False)
    print ('...done.')

    # resample and calculate resampled velocity
    # print ('Stage 2: Trajectory Resampling')
    # traj.reset_index(inplace=True, drop=True)
    # traj = gspp._resample_and_calculate_velocity_grouped(traj, velocity_window=3, velocity_drop_alpha=3, smoothing=True, res_rule = '60S', res_method='linear', crs = {'init': 'epsg:4326'}, drop_lon_lat = False, resampling_first=True, drop_outliers=False)

    # print ('Checkpoint 2: Saving Resampled Trajectories...')
    # if os.path.exists('./test_data/nari_dynamic_resampled.csv'):
    #     with open('./test_data/nari_dynamic_resampled.csv', 'a') as f:
    #         traj.to_csv(f, header=False, index=False)
    # else:
    #     print ('...creating file...\n')
    #     with open('./test_data/nari_dynamic_resampled.csv', 'w') as f:
    #         traj.to_csv(f, header=True, index=False)
    # print ('...done.')


    # upload to database
    print ('Stage 2: Uploading to PostgreSQL Database...')    
    for row in traj.itertuples(index=False):
        cur.execute(query_dynamic_ships_cleaned, (row.mmsi, row.turn, row.speed, row.course, row.heading, row.lon, row.lat, row.ts, row.velocity))
    con.commit()


cur.close()
con.close()