import os, sys
sys.path.append(os.path.join(os.path.expanduser('~')))
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
import pickle
from multiprocessing import cpu_count, Pool
from functools import partial
import datetime
import PyQt5
import matplotlib.pyplot as plt
import time

os.system("taskset -p 0xff %d" % os.getpid())

host    = '195.251.230.8'
db_name = 'doi105281zenodo1167595'
uname   = 'students'
pw      = 'infol@bdbl@bs2017'
port    = '46132'

# ts>1456802710 AND ts<1456975510

# traj_sql = 'SELECT * FROM ais_data.dynamic_ships WHERE  mmsi=226179000'
traj_sql = 'SELECT * FROM ais_data.dynamic_ships WHERE ts>1456702710 AND ts<1456975510'
ports_sql = 'SELECT * FROM ports.ports_of_brittany'
print('##########################')
print('Reading data from server')
con = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port = port)
trajectories = gpd.GeoDataFrame.from_postgis(traj_sql, con, geom_col='geom' )
#ports = gpd.GeoDataFrame.from_postgis(ports_sql, con, geom_col='geom' )

con.close()
print('Done reading data')

#ports.geom = ports.geom.apply(lambda x: x[0])

# SINGLE CORE
print('Cleaning...')
traj = gspp.clean_gdf(trajectories)
print('done')

pois_alpha=int(sys.argv[2])

# traj = pd.read_pickle(f'{str(sys.argv[1])}.pckl')
print('velocity')
traj = gspp.resample_and_calculate_velocity(traj, n_jobs=-1,velocity_window=100, res_rule='30S', drop_outliers=False, resampling_first=True)
print('done')
# traj.to_pickle(f'{str(sys.argv[2])}.pckl')
#traj = pd.read_pickle(f'{str(sys.argv[1])}.pckl')
ports = pd.read_pickle('ports.pckl')
start = time.time()
print('segmented')
traj = gspp.segment_trajectories(traj, pois_alpha=pois_alpha, pois_window=80, n_jobs=-1, np_split=True, feature='mmsi')
print(time.time()-start)
# print(traj.traj_id.unique())
# traj.to_pickle(f'{str(sys.argv[2])}.pckl')
traj.to_pickle(f'segmented_alpha_{pois_alpha}.pckl')
