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


def segment_resample_v2(vessel, port_bounds):                                               
	# def segment_resample_and_tag_v2(vessel, ports, port_radius=2000, rule = '60S', method='linear', crs = {'init': 'epsg:4326'}, drop_lon_lat = False, smoothing=False, window=15, center=False, pois_alpha=-1, pois_window=100, semantic=False):
	'''
	After the Segmentation Stage, for each sub-trajectory:
	  * we resample each trajectory
	  * calculate the velocity (per-point)
	  * we use our implementation on trajectory segmentation
		in order to add tags regarding the vessel's activity
	'''
	
	port_segmented_trajectories = gspp.segment_trajectories_v2(vessel, port_bounds)
	temporal_segmented_trajectories = gspp.__temporal_segment(port_segmented_trajectories, temporal_threshold=12)
	vessel_fn = pd.concat(temporal_segmented_trajectories, ignore_index=True)
	vessel_fn.sort_values('ts', inplace=True)
	vessel_fn.drop(['index'], axis=1, inplace=True)
	return vessel_fn


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
port_bounds = gspp.create_port_bounds(pd.read_pickle(os.path.join('.', 'ports.pckl')), epsg=2154, port_radius=2000)
mmsi_list_window_size = 2


### 2.8. Define the essential queries in order to fetch or upload data
traj_sql = 'SELECT * FROM ais_data.dynamic_ships_cleaned WHERE mmsi IN %s'

for i in tqdm(range(0, len(mmsi_list), mmsi_list_window_size)):
	# print (f'Preprocessing: {i+mmsi_list_window_size}/{len(mmsi_list)} Vessels...')
	mmsis = mmsi_list[i:i+mmsi_list_window_size]
	# print (f'Vessel MMSIs: {mmsis}')

	
	print ('Stage 1: Data Fetching')
	# usual stuff
	traj = pd.read_sql_query(traj_sql%(tuple(mmsis),), con=con)
	traj = gspp.gdf_from_df(traj, crs={'init': 'epsg:4326'})
	traj.drop(['id', 'status'], axis=1, inplace=True, errors='ignore')
	traj.sort_values('ts', inplace=True)
	traj.reset_index(inplace=True, drop=True)
	
	# calc correct velocities, acceleration and angle (some points were droped)
	print ('Stage 2: Calculating Time Outside Port')
	traj = traj.groupby('mmsi', group_keys=False).apply(segment_resample_v2, port_bounds)
	
	time_outside_port = traj.groupby(['mmsi', pd.to_datetime(traj.ts, unit='s').dt.date], group_keys=False).apply(lambda x: x.diff().sum()).to_frame().reset_index()
	print (time_outside_port)

	print ('Checkpoint 1: Saving Calculated AIS Activity Outside Port...')
	if os.path.exists(f'./test_data/ais_activity_outside_port_{CLUSTER_ID}.csv'):
		with open(f'./test_data/ais_activity_outside_port_{CLUSTER_ID}.csv', 'a') as f:
			time_outside_port.to_csv(f, header=False, index=False)
	else:
		print ('...creating file...\n')
		with open(f'./test_data/ais_activity_outside_port_{CLUSTER_ID}.csv', 'w') as f:
			time_outside_port.to_csv(f, header=True, index=False)
	print ('...done.')

	# print ('Checkpoint 2: Saving Calculated AIS Activity Within Port...')
	# if os.path.exists(f'./test_data/ais_activity_within_port_{CLUSTER_ID}.csv'):
	# 	with open(f'./test_data/ais_activity_within_port_{CLUSTER_ID}.csv', 'a') as f:
	# 		time_within_port.to_csv(f, header=False, index=False)
	# else:
	# 	print ('...creating file...\n')
	# 	with open(f'./test_data/ais_activity_within_port_{CLUSTER_ID}.csv', 'w') as f:
	# 		time_within_port.to_csv(f, header=True, index=False)
	# print ('...done.')
	
	break