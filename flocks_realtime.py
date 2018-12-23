import pandas as pd
import numpy as np
from haversine import haversine
import networkx as nx
from tqdm import tqdm_notebook as tqdm
import os, sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'Code'))
from lonelyboy.geospatial import preprocessing as gspp

df = pd.read_csv('4h_timeFrame_not_at_3_fing_am.csv')

def pairs_in_radius(df, diam=1000):
	'''
	Get all pairs with distance < diam
	'''
	res = []
	for ind_i, ind_j, val_i, val_j in nparray_combinations(df):
		dist = haversine(val_i, val_j)*1000
		if (dist<diam):
			res.append((ind_i,ind_j))   
	return res
			

def connected_edges(data, circular=True):
	'''
	Get circular (all points inside circle of diameter=diam) or density based (each pair with distance<diam)
	'''
	G = nx.Graph()
	G.add_edges_from(data)
	if circular:
		return [sorted(list(cluster)) for cluster in nx.find_cliques(G)]
	else:
		return [sorted(list(cluster)) for cluster in nx.connected_components(G)]


def nparray_combinations(arr):
	'''
	Get all combinations of points
	'''
	for i in range(arr.shape[0]):
		for j in range(i+1, arr.shape[0]):
			yield i, j, arr[i,:], arr[j,:]


def translate(sets, sdf):
	'''
	Get mmsis from clustered indexes
	'''
	return [sorted(tuple([sdf.iloc[point].mmsi for point in points])) for points in sets]


def get_circular_clusters(timeframe, diam):
	pairs = pairs_in_radius(timeframe[['lon', 'lat']].values, diam)
	return connected_edges(pairs)


def find_existing_flocks(x, present, past, last_ts):
	'''
	Find all clusters (present) that existed in the past (cluster subset of flock)
	'''
	# find the indices of past Dataframe where current cluster is subset of flock
	indcs = [set(x.clusters) < set(val) for val in past.loc[past.et==last_ts].clusters.values]
	# get the indices of the past dataframe where that occurs
	past.loc[(indcs)].index.tolist()


def replace_with_existing_flocks(x, present, to_keep, past):
	'''
	Replace current cluster with his existing flock
	'''
	if to_keep.iloc[x.name]:
		if len(past.iloc[to_keep.iloc[x.name]])>1:
			raise Exception('len > 1, something is wrong')

		x.dur = past.iloc[to_keep.iloc[x.name]].dur.values[0] +1
		x.st = past.iloc[to_keep.iloc[x.name]].st.values[0]
	return x


def get_current_clusters(sdf, ts, diam=1000):
	'''
	Get clusters and init them as a single flock
	'''
	present = pd.DataFrame([[tuple(val)] for (val) in translate(get_circular_clusters(sdf), sdf, diam)], columns=['clusters'])
	present['st'] = present['et'] = ts
	present['dur'] = 1
	return present


def present_is_subset_of_past(present, past, last_ts):
	'''
	Find and treat current clusters that exist in the past as a subset of a flock (used when flocks break apart to many smaller ones).
	'''
	to_keep = present.apply(find_existing_flocks, args=(present,past,last_ts,) , axis=1)

	present = present.apply(replace_with_existing_flocks, args=(present,to_keep,past,), axis=1)
	
	new = present.merge(past,on='clusters', how='left',suffixes=['','tmp'], indicator=True)
	new = new[new['_merge']=='left_only'].drop(['_merge'],axis=1).dropna(axis=1)
	
	return new


def past_is_subset_or_set_of_present(present, past, ts, last_ts):
	'''
	Find and propagate a flock if it's subset of a current cluster.
	'''
#     get if tuple of tmp1 is subset or equal of a row of tmp2
	to_keep = past.apply(lambda x: (True in [set(x.clusters) <= set(val) for val in present.clusters.values]) and (x.et == last_ts), axis=1)
	past.loc[to_keep,'et'] = ts
	past.loc[to_keep,'dur']= past.loc[to_keep].dur.apply(lambda x : x+1)
	return past


def get_flocks(new_clusters, clusters_to_keep):
	'''
	Concatenate all possible flocks to get the full flock dataframe
	'''
	return pd.concat([new_clusters,clusters_to_keep]).reset_index(drop=True)


if __name__=='__main__':

	# Parames
	diam = 1000
	min_samples = 10
	min_card = 2
	save_flag = False


	keep = pd.DataFrame()
	gspp.ts_from_str_datetime(df)

	for ind, (ts, sdf) in tqdm(enumerate(df.groupby('ts')), total=df.datetime.nunique()):
		
		present = get_current_clusters(sdf, ts, diam)
		
		# Init the first present as past
		if ind == 0:
			past = present
			last_ts = ts
			continue
			
		new_subsets = present_is_subset_of_past(present, past, last_ts)
		
		old_subsets_or_sets = past_is_subset_or_set_of_present(present, past, ts, last_ts)

		past = get_flocks(new_subsets, old_subsets_or_sets)

		# Only keep the entries that are either:
		# 1. Currently active -> (past.et==ts) 
		# or, 
		# 2. Been active for more that min_samples time steps -> (past.dur>min_samples).
		# and 
		# 3. Num of vessels in flock >= min_cardinality -> ([len(clst)>=min_card for clst in past.clusters])

		past = past.loc[((past.et==ts) | (past.dur>min_samples)) & ([len(clst)>=min_card for clst in past.clusters])] 
		last_ts = ts
	
	# keep this df and use it again as the db for the real time implementation
	if save_flag:
		past.to_csv('flocks.csv', index=False)

	