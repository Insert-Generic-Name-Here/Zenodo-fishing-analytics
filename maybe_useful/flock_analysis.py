import pandas as pd
import geopandas as gpd
import psycopg2
import os, sys
from shapely.geometry import Point
sys.path.append(os.path.join(os.path.expanduser('~'), 'Code'))
from lonelyboy.geospatial import preprocessing as gspp
import contextily as ctx
import glob
import numpy as np


def flock_analysis(traj, flocks):
    flocks['mean_vel'] = np.nan
    flocks['mean_dist'] = np.nan

    print('Analysis...')
    flocks['mean_vel'] = flocks.apply(lambda x: traj.loc[(traj.mmsi.isin(eval(x.clusters))) & (traj.ts >= x.st) & (traj.ts <= x.et)].velocity.mean(), axis=1)
    flocks['mean_dist'] = flocks.apply(lambda x: traj.loc[(traj.mmsi.isin(eval(x.clusters))) & (traj.ts >= x.st) & (traj.ts <= x.et)].velocity.mean()*((x.et-x.st)/60), axis=1)
    moving_flocks = flocks.loc[flocks.mean_vel>=1]
    non_moving_flocks = flocks.loc[flocks.mean_vel<1]
    
    print('### Results ###')
    print(f'From {len(flocks)} available flocks, {len(moving_flocks)} or {len(moving_flocks)/len(flocks)*100:.1f}% of them are moving and {len(non_moving_flocks)} or {len(non_moving_flocks)/len(flocks)*100:.1f}% are not moving\n')
    print(f'For the moving flocks, mean cardinality is {moving_flocks.clusters.apply(lambda x: len(eval(x))).mean():.1f} vessels \n\nFor the non moving flocks , mean cardinality is {non_moving_flocks.clusters.apply(lambda x: len(eval(x))).mean():.1f} vessels\n')
    print(f'For the moving flocks, mean distance traveled is {moving_flocks.mean_dist.mean():.1f} nautical miles and mean velocity is {moving_flocks.mean_vel.mean():.1f} knots\n')
    print('Done')
#     for _, flock in flocks.groupby('clusters'):
#         current_df = trajectories.loc[(trajectories.mmsi.isin(eval(flock.clusters))) & (trajectories.ts >= flock.st) & (trajectories.ts <= flock.et)]
     
print('Reading and analysing trajectories...\n')
traj = pd.read_csv('prepd_df.csv')   
traj.geom = traj[['lon', 'lat']].apply(lambda x: Point(x[0],x[1]), axis=1)
traj = gpd.GeoDataFrame(traj, geometry='geom')     


for flock_csv in glob.glob('flocks*.csv'):
    print(f'Reading {flock_csv}...\n')
    flocks = pd.read_csv(flock_csv)
    flock_analysis(traj, flocks)
