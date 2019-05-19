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
PORT_RADIUS = 2000 # Port Radius in Km


def create_port_bounds(ports, port_radius=2000):
	'''
	Given some Datapoints, create a circular bound of _port_radius_ kilometers.
	'''
	init_crs = ports.crs
	# We convert to EPSG-3310 is because the -euclidean- distance between two points is returned in meters.
	# So the buffer function creates a circle with radius _port_radius_ meters from the center point (i.e the port's location point).
	ports.loc[:, 'geom'] = ports.geom.to_crs(epsg=3310).buffer(port_radius).to_crs(init_crs)
	# After we create the ports bounding circle we convert back to its previous CRS.
	return ports


def get_port_popularity(ves_seg, ports):
    sindex = ves_seg.sindex

    port_popularity = pd.DataFrame([], columns=['port_id', '#arrivals_departures'])
    # find the points that intersect with each subpolygon and add them to points_within_geometry
    for (port_id, poly) in zip(ports.gid, ports.geom):
        # find approximate matches with r-tree, then precise matches from those approximate ones
        possible_matches_index = list(sindex.intersection(poly.bounds))
        possible_matches = ves_seg.iloc[possible_matches_index]
        precise_matches  = possible_matches[possible_matches.intersects(poly)]
        port_popularity  = port_popularity.append(pd.DataFrame([[port_id, len(precise_matches)]], columns=['port_id', '#arrivals_departures']))
        
    port_popularity.reset_index(inplace=True, drop=True)               
    return port_popularity


# Read MMSI list for each corresponding cluster
# mmsi_list = pd.read_pickle(os.path.join('.', 'data', 'pkl', 'mmsi_list.pckl'))[CLUSTER_ID]
mmsi_list = pd.read_pickle(os.path.join('.', 'mmsi_list.pckl'))[CLUSTER_ID]
mmsi_list_window_size = 10

# ports = pd.read_pickle(os.path.join('.', 'data', 'pkl', 'ports_raw.pkl'))
ports = pd.read_pickle(os.path.join('.', 'ports.pckl'))
ports = create_port_bounds(ports, port_radius=PORT_RADIUS)

port_popularity_report = pd.DataFrame([], columns=['port_id', '#arrivals_departures'])


for i in range(0, len(mmsi_list), mmsi_list_window_size):
    print (f'Preprocessing: {i+mmsi_list_window_size}/{len(mmsi_list)} Vessels...')
    mmsis = mmsi_list[i:i+mmsi_list_window_size]

    # Read MMSI CSV file to retrieve the records that correspond to the mmsi list
    print (f'\tInitial Stage: Fetching Records...')
    # csv_iter = gspp.read_csv_generator(os.path.join('.', 'data', 'csv', 'nari_dynamic.csv'), chunksize=50000, sep=',')
    csv_iter = gspp.read_csv_generator(os.path.join('.', 'data_mmsis.csv'), chunksize=50000, sep=',')
    chunk = pd.concat((chunk[chunk['mmsi'].isin(mmsis)] for chunk in csv_iter), ignore_index=True)    
    chunk = gspp.gdf_from_df(chunk, crs={'init':'epsg:4326'})

    # Clean Data (and save to PostgreSQL)
    print (f'\tStage 1: Cleaning Records...')
    chunk = gspp.clean_gdf(chunk)

    # Velocity Calculation & Trajectory Segmentation
    print (f'\tStage 2: Port Popularity...')
    if len(port_popularity_report) == 0:
        port_popularity_report = get_port_popularity(chunk, ports)
    else:
        port_popularity_report['#arrivals_departures'] = port_popularity_report['#arrivals_departures'].add(get_port_popularity(chunk, ports)['#arrivals_departures'], fill_value=0)

    print ('\n')
    # print (port_popularity_report.head(10))
    # if (i == 4):
    #     break
    
print ('\nSaving Result Data to CSV...')
port_popularity_report.to_csv(f'./test_data/port_popularity_report_{CLUSTER_ID}.csv', index=False, header=True)