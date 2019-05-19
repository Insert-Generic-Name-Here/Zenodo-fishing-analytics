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


def vessel_starts_and_docks_in_port(ves_seg, ports):
    sindex = ves_seg.sindex
    points_within_geometry = pd.DataFrame()
    for poly in ports.geom:
        # find approximate matches with r-tree, then precise matches from those approximate ones
        possible_matches_index = list(sindex.intersection(poly.bounds))
        possible_matches = ves_seg.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(poly)]
        points_within_geometry = points_within_geometry.append(precise_matches)
        
    points_within_geometry = points_within_geometry.drop_duplicates(subset=['mmsi', 'ts'])
    points_outside_geometry = ves_seg[~ves_seg.isin(points_within_geometry)].dropna(how='all')

    ves_seg.loc[:,'traj_id'] = np.nan
    # When we create the _traj_id_ column, we label each record with 0, 
	# if it's outside the port's radius and -1 if it's inside the port's radius. 
    ves_seg.loc[ves_seg.index.isin(points_within_geometry.index), 'traj_id'] = -1
    ves_seg.loc[ves_seg.index.isin(points_outside_geometry.index), 'traj_id'] = 0
    
    # print (ves_seg.iloc[[0,-1]])
    labels = ves_seg.iloc[[0,-1]].traj_id.unique().tolist()
    # print (labels)
    if len(labels) == 1 and labels[0] == -1 and len(ves_seg.loc[ves_seg.traj_id == 0]) != 0:
        tmp = ves_seg.loc[ves_seg.traj_id[ves_seg.traj_id.replace(-1,np.nan).ffill(limit=1).bfill(limit=1).notnull()].index]
        return True, len(tmp)
    else:
        return False, 0                   


# Read MMSI list for each corresponding cluster
# mmsi_list = pd.read_pickle(os.path.join('.', 'data', 'pkl', 'mmsi_list.pckl'))[CLUSTER_ID]
mmsi_list = pd.read_pickle(os.path.join('.', 'mmsi_list.pckl'))[CLUSTER_ID]
mmsi_list_window_size = 10
pre_segment_threshold = 12 # In Hours

# ports = pd.read_pickle(os.path.join('.', 'data', 'pkl', 'ports_raw.pkl'))
ports = pd.read_pickle(os.path.join('.', 'ports.pckl'))
ports = create_port_bounds(ports, port_radius=PORT_RADIUS)

mmsi_traj_report = pd.DataFrame(columns=["mmsi", "nr_of_AIS_signals", "nr_of_segments_type1", "nr_of_segments_type2", "nr_of_AIS_signals_in_segments_type2", "loss_ratio"])

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
    print (f'\tStage 2: Determining if an mmsi begins/ends from/to a port...')
    for idx, mmsi in tqdm(enumerate(mmsis), total=mmsi_list_window_size):
        # print (f'\tProcessed: {idx+1}/{mmsi_list_window_size} Vessels...\r', end='')
        good_traj_no = 0
        good_traj_ais_points = 0

        vessel = chunk.loc[chunk.mmsi==mmsi].reset_index(drop=True)
        break_points = vessel.ts.diff(-1).abs().index[vessel.ts.diff()>60*60*pre_segment_threshold]
        if (len(break_points) > 0):
            dfs = np.split(vessel, break_points)
        else:
            dfs = [vessel]
            
        for i in range(0,len(dfs)):        
            vessel_docks, vessel_type2_pts = vessel_starts_and_docks_in_port(dfs[i], ports)
            if (vessel_docks):
                good_traj_no += 1
                good_traj_ais_points += vessel_type2_pts

        mmsi_traj_report = mmsi_traj_report.append({"mmsi":mmsi, "nr_of_AIS_signals":len(vessel), "nr_of_segments_type1":len(break_points)+1, "nr_of_segments_type2":good_traj_no, "nr_of_AIS_signals_in_segments_type2":good_traj_ais_points, "loss_ratio":np.around(good_traj_ais_points/len(vessel), decimals=2)}, ignore_index=True)

    print ('\n')
    
print ('\nSaving Result Data to CSV...')
mmsi_traj_report.to_csv(f'./test_data/mmsi_traj_report_v2_{CLUSTER_ID}_with_ratio.csv', index=False, header=True)