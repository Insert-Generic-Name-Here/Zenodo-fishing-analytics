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
sys.path.append(os.path.join(os.path.expanduser('~/Documents/Insert-Generic-Name-Here')))
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


# **#4** Get Trip Contributions per GP
def classify_trips(df):
    df2 = pd.Series([0], index=['class'])
    
    if df['label'].iloc[0] == -1 and df['label'].iloc[-1] == -1:
        df2['class'] = 1
    elif df['label'].iloc[0] == -1 and df['label'].iloc[-1] == 0:
        df2['class'] += 2
    elif df['label'].iloc[0] == 0 and df['label'].iloc[-1] == -1:
        df2['class'] += 3
    elif df['label'].iloc[0] == 0 and df['label'].iloc[-1] == 0:
        df2['class'] += 4
    
    return df2


def get_gp_trip_contributions_parallel(y):
    print ('Connecting to Database...')
    con = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port=port)
    print ('Connected to Database!')
    
    df_stat4 = []
    for row in tqdm(y.itertuples(), total=len(y)):
        df_stat4_row = pd.DataFrame([{'GP':row.clusters, 'C1':0, 'C2':0, 'C3':0, 'C4':0}], columns=['GP', 'C1', 'C2', 'C3', 'C4'])

        row_dynamic = pd.read_sql_query(query%(tuple(row.clusters), row.st, row.et), con=con)
        row_dynamic_trips = row_dynamic.groupby(['mmsi', 'trip_id']).apply(lambda x: CLASSIFIED_TRIPS.loc[(CLASSIFIED_TRIPS.mmsi == x.name[0]) & (CLASSIFIED_TRIPS.trip_id == x.name[1])]['class'].values).to_frame()

        for trip_contr in row_dynamic_trips[0].value_counts().iteritems():
            df_stat4_row.iloc[:, trip_contr[0][0]] = trip_contr[1]

        df_stat4.append(df_stat4_row)
    con.close()
    return pd.concat(df_stat4, ignore_index=True)



### 2. Importing the Server Credentials & Connectiing to Server
properties = configparser.ConfigParser()
properties.read(os.path.join('..','sql_server.ini'))
properties = properties['SERVER']

host    = properties['host']
db_name = properties['db_name']
uname   = properties['uname']
pw      = properties['pw']
port    = properties['port']

# con = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port=port)
query = 'SELECT * FROM ais_data.dynamic_ships_min_trip_card_3_segmented_12h_resampled_1min_v2 WHERE mmsi IN %s AND datetime BETWEEN \'%s\' AND \'%s\';'



## READ DATA CSV
# <FILENAME, GP TYPE,  AVG. DISTANCE>
ROWS = []
gp_runs_csv_path = './data/csv/GROUP_PATTERNS/'



csv_dir = 'data/csv/nari_dynamic_min_trip_card_3_no_resampling_correcred_bug_V2'
# csv_dir = 'test_data/nari_dynamic_min_trip_card_3_no_resampling_correcred_bug_V2'
df_trips = []
for file in tqdm(os.listdir(csv_dir)):
    df_dynamic_trips = pd.read_csv(os.path.join(csv_dir, file))
    df_trip = df_dynamic_trips.groupby(['mmsi', 'trip_id'], group_keys=False).apply(lambda df_dt: classify_trips(df_dt)).reset_index()
    df_trips.append(df_trip)
CLASSIFIED_TRIPS = pd.concat(df_trips)



for filepath in tqdm(os.listdir(gp_runs_csv_path)[0:3]):
    print(filepath)
    if not ('.csv' in filepath):
        continue

    df = pd.read_csv(os.path.join(gp_runs_csv_path, filepath))

    if len(df) == 0:
        print('No GPs Here!')
        continue

    df.loc[:,'clusters'] = df.clusters.apply(tuple_str_to_list)

    # SAVE RESULTS
    df_RESULTS = parallelize_dataframe(df, get_gp_trip_contributions_parallel)
    df_RESULTS.to_csv(os.path.join(gp_runs_csv_path, f'{filepath.split(".")[0]}_trip_contributions.csv'), index=False, header=True)
