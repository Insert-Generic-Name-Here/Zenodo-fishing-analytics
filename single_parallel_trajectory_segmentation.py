import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.preprocessing import MinMaxScaler
import psycopg2
from random import choice
import contextily as ctx
import numpy as np
from shapely.geometry import Point, LineString, shape
from tqdm import tqdm_notebook
import numpy as np
from multiprocessing import cpu_count, Pool
from functools import partial
import datetime
from lonelyboy.geospatial import plots as gsplt
from lonelyboy.geospatial import preprocessing as gspp
from lonelyboy.timeseries import lbtimeseries as tspp
import configparser, os
import time

pd.options.mode.chained_assignment = None  # default='warn'

host    = '195.251.230.8'
db_name = 'doi105281zenodo1167595'
uname   = 'students'
pw      = 'infol@bdbl@bs2017'
port    = '46132'

# ts>1456802710 AND ts<1456975510 

# traj_sql = 'SELECT * FROM ais_data.dynamic_ships WHERE  mmsi=226179000'
traj_sql = 'SELECT * FROM ais_data.dynamic_ships WHERE  ts>1456702710 AND ts<1456975510 '
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
traj = trajectories.copy()
print('started single core processing')
start_single = time.time()
traj = gspp.clean_gdf(traj)
single_final = gspp.segment_trajectories(traj)
print (f'Single core took {time.time()-start_single} secs.')

print(f'Started multi core processing with {cpu_count()} cores')
traj = trajectories.copy()
start_mult = time.time()
traj = gspp.parallelize_dataframe(traj, gspp.clean_gdf, np_split=False, num_partitions=cpu_count())
multi_final = gspp.parallelize_dataframe(traj, gspp.segment_trajectories, np_split=False, num_partitions=cpu_count())

print (f'Multi core took {time.time()-start_mult} secs.')

print (f'Result -> {single_final.equals(multi_final)}')
multi_final.to_csv('multifinal.csv')
single_final.to_csv('singlefinal.csv')
