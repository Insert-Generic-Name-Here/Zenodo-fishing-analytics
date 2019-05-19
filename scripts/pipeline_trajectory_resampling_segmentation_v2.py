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
ports     = pd.read_pickle(os.path.join('.', 'ports.pckl'))
mmsi_list = pd.read_pickle(os.path.join('.', 'mmsi_list.pckl'))[CLUSTER_ID]
mmsi_list_window_size = 10


### 2.8. Define the essential queries in order to fetch or upload data
traj_sql = 'SELECT * FROM ais_data.dynamic_ships_cleaned WHERE mmsi IN %s'
query_dynamic_ships_resampled_segmented = "INSERT INTO ais_data.dynamic_ships_segmented_12h_resampled_1min (mmsi, turn, speed, course, heading, lon, lat, ts, datetime, traj_id, traj_id_temporal_gap, real_point, velocity) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"

for i in tqdm(range(0, len(mmsi_list), mmsi_list_window_size)):
    # print (f'Preprocessing: {i+mmsi_list_window_size}/{len(mmsi_list)} Vessels...')
    mmsis = mmsi_list[i:i+mmsi_list_window_size]
    # print (f'Vessel MMSIs: {mmsis}')

    print ('Initial Stage: Data Fetching')
    traj = pd.read_sql_query(traj_sql%(tuple(mmsis),), con=con)
    traj = gspp.gdf_from_df(traj, crs={'init': 'epsg:4326'})

    # resample and segment trajectories
    print ('Stage 1: Trajectory Resampling and Segmentation')
    for _, vessel in traj.groupby('mmsi'):
        vessel = gspp.segment_resample_v2(vessel, ports, port_epsg=2154, port_radius=2000, temporal_threshold=12, rule = '60S', method='linear', crs = {'init': 'epsg:4326'}, drop_lon_lat = False, smoothing=False, window=15, center=False)
        vessel.rename(columns={'traj_id_12h_gap': 'traj_id_temporal_gap'}, inplace=True)

        print ('Checkpoint 1: Saving Resampled and Segmented Trajectories...')
        if os.path.exists(f'./test_data/nari_dynamic_resampled_and_segmented_{CLUSTER_ID}.csv'):
            with open(f'./test_data/nari_dynamic_resampled_and_segmented_{CLUSTER_ID}.csv', 'a') as f:
                vessel.to_csv(f, header=False, index=False)
        else:
            print ('...creating file...\n')
            with open(f'./test_data/nari_dynamic_resampled_and_segmented_{CLUSTER_ID}.csv', 'w') as f:
                vessel.to_csv(f, header=True, index=False)
        print ('...done.')


        # upload to database
        print ('Stage 2: Uploading to PostgreSQL Database...')    
        for row in vessel.itertuples(index=False):
            cur.execute(query_dynamic_ships_resampled_segmented, (row.mmsi, row.turn, row.speed, row.course, row.heading, row.lon, row.lat, row.ts, row.datetime, row.traj_id, row.traj_id_temporal_gap, row.real_point, row.velocity))
        con.commit()

cur.close()
con.close()