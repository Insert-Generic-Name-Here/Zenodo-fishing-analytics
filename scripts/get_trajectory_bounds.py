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
from tqdm import tqdm
import pandas as pd
import numpy as np


CLUSTER_ID = int(sys.argv[1]) # 0 = MASTER; 1-5 = SLAVES

# Read MMSI list for each corresponding cluster
mmsi_list = pd.read_pickle(os.path.join('.', 'mmsi_list.pckl'))[CLUSTER_ID]
mmsi_list_window_size = 20

# ports = pd.read_pickle(os.path.join('.', 'data', 'pkl', 'ports_raw.pkl'))
mmsi_traj_bounds = pd.DataFrame(columns=["mmsi","min_lat", "min_lon", "max_lat", "max_lon"])

for i in range(0, len(mmsi_list), mmsi_list_window_size):
    print (f'Preprocessing: {i+mmsi_list_window_size}/{len(mmsi_list)} Vessels...')
    mmsis = mmsi_list[i:i+mmsi_list_window_size]

    # Read MMSI CSV file to retrieve the records that correspond to the mmsi list
    print (f'\tInitial Stage: Fetching Records...')
    # csv_iter = gspp.read_csv_generator(os.path.join('.', 'data', 'csv', 'nari_dynamic.csv'), chunksize=50000, sep=',')
    csv_iter = gspp.read_csv_generator(os.path.join('.', 'data_mmsis.csv'), chunksize=150000, sep=',')
    chunk = pd.concat((chunk[chunk['mmsi'].isin(mmsis)] for chunk in csv_iter), ignore_index=True)    
    chunk = gspp.gdf_from_df(chunk, crs={'init':'epsg:4326'})

    # Clean Data (and save to PostgreSQL)
    print (f'\tStage 1: Cleaning Records...')
    chunk = gspp.clean_gdf(chunk)

    # Velocity Calculation & Trajectory Segmentation
    print (f'\tStage 2: Determining if an mmsi begins/ends from/to a port...')
    for idx, mmsi in tqdm(enumerate(mmsis), total=mmsi_list_window_size):
        vessel = chunk.loc[chunk.mmsi==mmsi].reset_index(drop=True)
        mmsi_traj_bounds = mmsi_traj_bounds.append({"mmsi":mmsi, "min_lat":vessel.geom.x.min(), "min_lon":vessel.geom.y.min(), "max_lat":vessel.geom.x.max(), "max_lon":vessel.geom.y.max()}, ignore_index=True)

    print ('\n')
    
print ('\nSaving Result Data to CSV...')
mmsi_traj_bounds.to_csv(f'./test_data/mmsi_traj_bounds_{CLUSTER_ID}.csv', index=False, header=True)