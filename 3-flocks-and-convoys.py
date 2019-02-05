import json
import os, sys
#sys.path.append(os.path.join(os.path.expanduser('~'), 'Documents/Insert-Generic-Name-Here/'))
# sys.path

from lonelyboy.geospatial import plots as gsplt
from lonelyboy.geospatial import preprocessing as gspp
from lonelyboy.timeseries import lbtimeseries as tspp
from lonelyboy.geospatial import group_patterns as gsgp

import psycopg2
import numpy as np
import configparser
import pandas as pd
import geopandas as gpd
import contextily as ctx
from random import choice
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans, MeanShift
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from shapely.geometry import Point, LineString, shape
from haversine import haversine
from datetime import datetime, timedelta
from tqdm import tqdm
tqdm.pandas()

from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans, MeanShift
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from haversine import haversine
from datetime import datetime, timedelta


def dbshit(X,scaler,colname,**dbscan_params):
	X.reset_index(inplace=True)
	#     print (X)
	X[colname] = DBSCAN(**dbscan_params).fit(scaler.transform(X[['lon','lat']])).labels_
	#     X['labels'] = se.values
	return X

if os.path.isfile('data/prepd_timeframe.csv'):
	sample_timeFrame = pd.read_csv('data/prepd_timeframe.csv', index_col=0)
else:
	host    = '195.251.230.8'
	db_name = 'doi105281zenodo1167595'
	uname   = 'students'
	pw      = 'infol@bdbl@bs2017'
	port    = '46132'

	traj_sql = 'SELECT * FROM ais_data.dynamic_ships WHERE ts>1456802710 AND ts<1456975510  '
	ports_sql = 'SELECT * FROM ports.ports_of_brittany'

	print('Connecting to server')
	con = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port = port)

	sample_trajectories = gpd.GeoDataFrame.from_postgis(traj_sql, con, geom_col='geom' )

	# ports = gpd.GeoDataFrame.from_postgis(ports_sql, con, geom_col='geom' )
	# ports.geom = ports.geom.apply(lambda x: x[0])

	# print(f'Fetched {sizeof_fmt(ports.memory_usage().sum())}')

	con.close()
	print ('Processing...')
	sample_trajectories = sample_trajectories.drop_duplicates(subset=['mmsi', 'ts']).sort_values('ts').reset_index(drop=True)
	sample_trajectories['velocity'] = np.nan
	sample_trajectories = sample_trajectories.groupby(['mmsi'], as_index=False).apply(gspp.calculate_velocity, smoothing=False, window=15, center=False).reset_index(drop=True)

	### DENOISE SAMPLE_TRAJECTORIES BASED ON A VELOCITY THRESHOLD (POTENTIAL-AREA-OF-ACTIVITY)
	sample_trajectories = gspp.PotentialAreaOfActivity(sample_trajectories, velocity_threshold = 102.2)

	sample_trajectories_resampled = sample_trajectories.groupby(['mmsi'], as_index=False).apply(gspp.resample_geospatial, rule = '60S', method='linear', crs = {'init': 'epsg:4326'}, drop_lon_lat = False).reset_index(drop=True) 
	sample_trajectories_resampled.sort_values(['datetime'], ascending=True, inplace=True)

	timeWindow = sample_trajectories_resampled.datetime.unique()[0]
	timeWindow = [timeWindow + np.timedelta64(60*i, 's') for i in range(1, 121)] #2H
	# timeWindow = [timeWindow + np.timedelta64(60*i, 's') for i in range(1, 61)] #1H
	# timeWindow = [timeWindow + np.timedelta64(60*i, 's') for i in range(1, 31)] #30MIN
	sample_timeFrame = sample_trajectories_resampled.loc[sample_trajectories_resampled.datetime.isin(timeWindow)].sort_values('datetime').reset_index(drop=True)
	sample_timeFrame.to_csv('data/prepd_timeframe.csv')

#sample_timeFrame = pd.read_csv('sample_timeFrame.csv')

sample_timeFrame2 = sample_timeFrame[['mmsi', 'lon', 'lat', 'datetime']]

print('Scaling data...')

scaler1 = MinMaxScaler(feature_range=(0,1))
scaler1.fit(sample_timeFrame2[['lat','lon']].values)


if not len(sys.argv) > 1:
	with open('dbscan_sims.json', 'r') as fp:
			dbscan_params_list = json.load(fp)

	print ('Clustering...')

	for i, colname in enumerate(dbscan_params_list):
		sample_timeFrame2.loc[:,colname] = np.nan

		sample_timeFrame2 = sample_timeFrame2.groupby(['datetime'], as_index=False).apply(dbshit, scaler1, colname, **dbscan_params_list[colname]).reset_index(drop=True).drop(['index'], axis=1)
		print(f'{i+1} out of {len(dbscan_params_list)} done...')
else:
	dbscan_params = {}

	for params in sys.argv[1:]:
		dbscan_params[params.split('=')[0]] = float(params.split('=')[1])

		colname='labels'
		sample_timeFrame2.loc[:,colname] = np.nan
		sample_timeFrame2 = sample_timeFrame2.groupby(['datetime'], as_index=False).apply(dbshit, scaler1, colname, **dbscan_params).reset_index(drop=True).drop(['index'], axis=1)
print('Saving...')
sample_timeFrame2.to_csv(f'scaled_clustered_tf.csv')
print ('Done')
