import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from random import choice
pd.options.mode.chained_assignment = None 
import configparser
import pickle
import psycopg2
import psycopg2.extras
import contextily as ctx

import os, sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'Documents', 'Insert-Generic-Name-Here'))
# sys.path

from lonelyboy.geospatial import plots as gsplt
from lonelyboy.geospatial import preprocessing as gspp
from lonelyboy.timeseries import lbtimeseries as tspp

from tqdm import tqdm
import multiprocessing



def tuple_str_to_tuple(_str_):
    return tuple(map(int, _str_[1:-1].split(',')))

def tuple_str_to_list(_str_):
    return list(map(int, _str_[1:-1].split(',')))


def parallelize_dataframe(df_par, func):
    num_cores = multiprocessing.cpu_count()-1  #leave one free to not freeze machine
    num_partitions = num_cores #number of partitions to split dataframe
    
    df_split = np.array_split(df, num_partitions)
    pool = multiprocessing.Pool(num_cores)
    
    df_res = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df_res


# **#2** Distribution of Group Patterns Size (Flocks)
def get_gp_avg_velocity(x, conn):
    gp = pd.read_sql_query(query%(tuple(x.clusters), x.st, x.et), con=conn)
    return gp.groupby('mmsi', group_keys=False).apply(lambda df: df.velocity.mean()).mean()

def get_gp_avg_velocity_parallel(y):
    print ('Connecting to Database...')
    con = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port=port)
    print ('Connected to Database!')
    for row in tqdm(y.itertuples(), total=len(y)):
        y.loc[row.Index, 'speed'] = get_gp_avg_velocity(row, con)
    con.close()
    return y


# **#3** Total Distance Travelled per Group Pattern (Flocks)
def get_gp_travelled_distance(x, conn):
    gp = pd.read_sql_query(query%(tuple(x.clusters), x.st, x.et), con=conn)
    center_point = gp.groupby('datetime', group_keys=False, as_index=False).apply(lambda df: pd.Series({'mean_lon': df.lon.mean(), 'mean_lat': df.lat.mean()}))
    
    current_loc = center_point[['mean_lon', 'mean_lat']]
    next_loc = center_point[['mean_lon', 'mean_lat']].shift(-1)
    return np.nansum([gspp.haversine(tuple(curr_loc), tuple(nxt_loc))  for curr_loc, nxt_loc in zip(current_loc.values.tolist(), next_loc.values.tolist())])
                            

def get_gp_travelled_distance_parallel(y):
    print ('Connecting to Database...')
    con = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port=port)
    print ('Connected to Database!')
    for row in tqdm(y.itertuples(), total=len(y)):
        y.loc[row.Index, 'distance'] = get_gp_travelled_distance(row, con)
    con.close()
    return y



properties = configparser.ConfigParser()
properties.read(os.path.join('.','sql_server.ini'))
properties = properties['SERVER']

host    = properties['host']
db_name = properties['db_name']
uname   = properties['uname']
pw      = properties['pw']
port    = properties['port']
con     = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port=port)


gp_runs_csv_path = './data/csv/GROUP_PATTERNS_VAR_SR'
save_path = './data'


## READ DATA CSV
# <FILENAME, GP TYPE, #GPs, AVG. DURATION, AVG.SIZE, AVG.VELOCITY, AVG. DISTANCE>
ROWS = []


for filepath in tqdm(os.listdir(gp_runs_csv_path)):
    query = f'SELECT * FROM ais_data.dynamic_ships_min_trip_card_3_segmented_12h_resampled_{int(filepath.split("_")[1][:-1])//60}min_v2 WHERE mmsi IN %s AND datetime BETWEEN \'%s\' AND \'%s\';'

    print(filepath)
    if not ('.csv' in filepath):
        continue
    
    df = pd.read_csv(os.path.join(gp_runs_csv_path, filepath))
    gp_type = filepath.split('_')[0]
    no_of_gps = len(df)

    if len(df) == 0:
        print('No GPs Here!')
        NEW_ROW = [filepath.split('.')[0], gp_type, 0, 0, 0, 0, 0]
        ROWS.append(NEW_ROW)
        continue



    # **#1** Average Duration per Group Pattern
    df.loc[:, 'duration'] = (df.et.apply(pd.to_datetime, 's') - df.st.apply(pd.to_datetime, 's')).apply(lambda x: x.seconds//60)

    # **#2** Distribution of Group Patterns Size (Flocks)
    df.loc[:, 'size'] = df.clusters.apply(tuple_str_to_list).apply(len)

    # **#3** Avg Velocity per Group Pattern (Flocks)
    df.loc[:,'clusters'] = df.clusters.apply(tuple_str_to_list)
    avg_duration = df.loc[:, 'duration'].mean()
    avg_size = df.loc[:, 'size'].mean()


    df = parallelize_dataframe(df, get_gp_avg_velocity_parallel)
    print (df.columns)
    avg_velocity = df.loc[:, 'speed'].mean()
    
    df = parallelize_dataframe(df, get_gp_travelled_distance_parallel)   
    print (df.columns)
    avg_distance = df.loc[:, 'distance'].mean()
   

    NEW_ROW = [filepath.split('.')[0], gp_type, no_of_gps, avg_duration, avg_size, avg_velocity, avg_distance]
    ROWS.append(NEW_ROW)

    print(NEW_ROW)


# SAVE RESULTS
STATS_DF = pd.DataFrame(ROWS, columns=['FILENAME', 'GP.TYPE', '#GPs', 'AVG.DURATION', 'AVG.SIZE', 'AVG.VELOCITY', 'AVG.DISTANCE'])
STATS_DF.to_csv(os.path.join(save_path, 'GP_STATS_VAR_SR.csv'), index=False, header=True)


con.close()
