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
mmsis = [305166000, 228037600, 205067000, 227519920]

### 2.8. Define the essential queries in order to fetch or upload data
traj_sql = 'SELECT * FROM ais_data.dynamic_ships_cleaned WHERE mmsi IN %s'
   
print ('Initial Stage: Data Fetching')
traj = pd.read_sql_query(traj_sql%(tuple(mmsis),), con=con)
traj = gspp.gdf_from_df(traj, crs={'init': 'epsg:4326'})


print ('Checkpoint 1: Saving Test Trajectories...')
with open(f'./test_data/nari_dynamic_test.csv', 'w') as f:
    traj.to_csv(f, header=True, index=False)
print ('...done.')

cur.close()
con.close()