import os, sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'Documents/Insert-Generic-Name-Here/'))
# sys.path

from lonelyboy.geospatial import plots as gsplt
from lonelyboy.geospatial import preprocessing as gspp
from lonelyboy.timeseries import lbtimeseries as tspp
from lonelyboy.geospatial import group_patterns as gsgp

import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
tqdm.pandas()

from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans, MeanShift
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from haversine import haversine
from datetime import datetime, timedelta


sample_timeFrame = pd.read_csv('sample_timeFrame.csv')
print (sample_timeFrame.head(10))

sample_timeFrame2 = sample_timeFrame[['mmsi', 'heading', 'lon', 'lat', 'datetime']]

dbscan_params = {'eps':2.5, 
                 'min_samples':3, 
                 'metric':haversine,
                 'n_jobs':-1}


print ('NOW I BEGIN')
sample_timeFrame2['cluster_idx'] = sample_timeFrame2.groupby('datetime', as_index=False)[['mmsi', 'lon', 'lat']].progress_apply(lambda X: DBSCAN(**dbscan_params).fit(X[['lon','lat']].values).labels_)
print (sample_timeFrame2.head(10))
print ('NOW I FINISH')