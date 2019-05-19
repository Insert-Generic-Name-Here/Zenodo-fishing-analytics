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


def calculate_acceleration(gdf):
	'''
	Return given dataframe with an extra acceleration column that
	is calculated using the distance covered in a given amount of time.
	TODO: use the get distance method to save some space
	'''
	# if there is only one point in the trajectory its velocity will be the one measured from the speedometer
	if len(gdf) == 1:
		gdf['acceleration'] = 0
		return gdf

	gdf['acceleration'] = gdf.velocity.diff().divide(gdf.ts.diff())
	# gdf['acceleration'] = gdf.velocity.diff(-1)

	gdf = gdf.loc[gdf['mmsi'] != 0]
	gdf.dropna(subset=['mmsi', 'geom'], inplace=True)

	return gdf.fillna(0)


def calculate_course(point1, point2):
    '''
        Calculating initial bearing between two points
        (see http://www.movable-type.co.uk/scripts/latlong.html)
    '''
    lon1, lat1 = point1[0], point1[1]
    lon2, lat2 = point2[0], point2[1]

    dlat = (lat2 - lat1)
    dlon = (lon2 - lon1)
    numerator = np.sin(dlon) * np.cos(lat2)
    denominator = (
        np.cos(lat1) * np.sin(lat2) -
        (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    )

    theta = np.arctan2(numerator, denominator)
    theta_deg = (np.degrees(theta) + 360) % 360
    return theta_deg


def calculate_bearing(gdf):
	'''
	Return given dataframe with an extra velocity column that is calculated using the distance covered in a given amount of time
	TODO - use the get distance method to save some space
	'''
	# if there is only one point in the trajectory its velocity will be the one measured from the speedometer
	if len(gdf) == 1:
		gdf['bearing'] = np.nan
		return gdf

	# create columns for current and next location. Drop the last columns that contains the nan value
	gdf['current_loc'] = gdf.geom.apply(lambda x: (x.x,x.y))
	gdf['next_loc'] = gdf.geom.shift(-1)
	gdf = gdf[:-1]
	gdf.loc[:,'next_loc'] = gdf.next_loc.apply(lambda x : (x.x,x.y))
	# get the distance traveled in n-miles and multiply by the rate given (3600/secs for knots)
	gdf.loc[:,'bearing'] = gdf[['current_loc', 'next_loc']].apply(lambda x: calculate_course(x[0], x[1]), axis=1).bfill().ffill()

	gdf.drop(['current_loc', 'next_loc'], axis=1, inplace=True)
	gdf = gdf.loc[gdf['mmsi'] != 0]
	gdf.dropna(subset=['mmsi', 'geom'], inplace=True)

	return gdf


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
	
	# calc correct velocities, acceleration and angle (some points were droped)
	print ('Stage 2: Calculating Velocity')
	traj = traj.groupby('mmsi', group_keys=False).apply(gspp.calculate_velocity, smoothing=False)
	print ('Stage 3: Calculating Acceleration')
	traj = traj.groupby('mmsi', group_keys=False).apply(calculate_acceleration)
	print ('Stage 4: Calculating Bearing')
	traj = traj.groupby('mmsi', group_keys=False).apply(calculate_bearing)
	
	print ('Checkpoint 1: Saving Calculated Trajectory Features...')
	if os.path.exists(f'./test_data/nari_dynamic_vanilla_features_{CLUSTER_ID}_no_smoothing.csv'):
		with open(f'./test_data/nari_dynamic_vanilla_features_{CLUSTER_ID}_no_smoothing.csv', 'a') as f:
			traj.to_csv(f, header=False, index=False)
	else:
		print ('...creating file...\n')
		with open(f'./test_data/nari_dynamic_vanilla_features_{CLUSTER_ID}_no_smoothing.csv', 'w') as f:
			traj.to_csv(f, header=True, index=False)
	print ('...done.')