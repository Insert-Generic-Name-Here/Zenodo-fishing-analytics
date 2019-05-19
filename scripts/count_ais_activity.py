import os, sys
# sys.path.append(os.path.join(os.path.expanduser('~'), 'Documents/Insert-Generic-Name-Here'))
sys.path.append(os.path.join(os.path.expanduser('~')))

from lonelyboy.geospatial import plots as gsplt
from lonelyboy.geospatial import preprocessing as gspp
from lonelyboy.timeseries import lbtimeseries as tspp
from lonelyboy.geospatial import group_patterns_v2 as gsgp

from shapely.geometry import Point, LineString, shape
import matplotlib.pyplot as plt # Importing Libraries
import geopandas as gpd
import contextily as ctx
import configparser
from tqdm import tqdm
import pandas as pd
import numpy as np
import psycopg2


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


### 2.5. Read MMSI list for each corresponding cluster and set up the MMSI Window Size
mmsi_list = pd.read_pickle(os.path.join('.', 'mmsi_list.pckl'))[CLUSTER_ID]
mmsi_list_window_size = 10

### 2.8. Define the essential queries in order to fetch or upload data
traj_sql = 'SELECT * FROM ais_data.dynamic_ships_segmented_12h_resampled_1min WHERE mmsi IN %s'

for i in tqdm(range(0, len(mmsi_list), mmsi_list_window_size)):
    mmsis = mmsi_list[i:i+mmsi_list_window_size]

    # Read MMSI CSV file to retrieve the records that correspond to the mmsi list
    print (f'\tInitial Stage: Fetching Records...')
    traj = pd.read_sql_query(traj_sql%(tuple(mmsis),), con=con)
    traj.sort_values('ts', inplace=True)

    if (len(traj) == 0):
        continue

    print (f'\tStage 1: Calculating Time(h) Outside Port...')
    ais_activity_outside_port = traj.groupby(['mmsi', 'traj_id_temporal_gap', pd.to_datetime(traj.ts, unit='s').dt.date], group_keys=False).apply(lambda df: df.ts.diff().sum()/3600).to_frame().reset_index()
    ais_activity_outside_port.columns = ['mmsi', 'traj_id_temporal_gap','date', '#hrs']
    ais_activity_outside_port = ais_activity_outside_port.groupby(['mmsi', 'date'], group_keys=False).apply(lambda df: np.around(df['#hrs'].sum(), 3)).to_frame().reset_index()
    ais_activity_outside_port.columns = ['mmsi', 'date', '#hrs']
    

    print (f'\tStage 2: Calculating Time(h) Within Port...')
    ais_activity_within_port = pd.DataFrame([], columns=['mmsi', 'date', '#hrs'])
    for groupby_cols, mmsi_traj in traj.groupby(['mmsi', pd.to_datetime(traj.ts, unit='s').dt.date]):
        breaking_points = mmsi_traj.loc[mmsi_traj.traj_id.diff() == 1].index.tolist()
        if (len(breaking_points) == 0):
            ais_activity_within_port = ais_activity_within_port.append(pd.DataFrame([[groupby_cols[0], groupby_cols[1], 0]], columns=['mmsi', 'date', '#hrs']))
            continue
        hrs = np.around(np.sum([mmsi_traj.loc[bp-1:bp,:].ts.diff().values[1] for bp in breaking_points])/3600, 3)
        ais_activity_within_port = ais_activity_within_port.append(pd.DataFrame([[groupby_cols[0], groupby_cols[1], hrs]], columns=['mmsi', 'date', '#hrs']))
    ais_activity_within_port.reset_index(drop=True, inplace=True)


    print ('Checkpoint 1: Saving AIS Activity Outside Port...')
    if os.path.exists(f'./test_data/nari_dynamic_ais_activity_report_outside_port_{CLUSTER_ID}.csv'):
        with open(f'./test_data/nari_dynamic_ais_activity_report_outside_port_{CLUSTER_ID}.csv', 'a') as f:
            ais_activity_outside_port.to_csv(f, header=False, index=False)
    else:
        print ('...creating file...\n')
        with open(f'./test_data/nari_dynamic_ais_activity_report_outside_port_{CLUSTER_ID}.csv', 'w') as f:
            ais_activity_outside_port.to_csv(f, header=True, index=False)
    print ('...done.')

    print ('Checkpoint 2: Saving AIS Activity Within Port...')
    if os.path.exists(f'./test_data/nari_dynamic_ais_activity_report_within_port_{CLUSTER_ID}.csv'):
        with open(f'./test_data/nari_dynamic_ais_activity_report_within_port_{CLUSTER_ID}.csv', 'a') as f:
            ais_activity_within_port.to_csv(f, header=False, index=False)
    else:
        print ('...creating file...\n')
        with open(f'./test_data/nari_dynamic_ais_activity_report_within_port_{CLUSTER_ID}.csv', 'w') as f:
            ais_activity_within_port.to_csv(f, header=True, index=False)
    print ('...done.')
    print ('\n')

con.close()