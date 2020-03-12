import os
import h5py
import json
import numpy as np
from numpy import linalg as la
from scipy.linalg import eigh as largest_eigh
from dppy.finite_dpps import FiniteDPP
#Upgrade from v0.19.0 to v0.22.0
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from numpy.random import rand, randn, RandomState
from dppy.finite_dpps import FiniteDPP
from dppy.utils import example_eval_L_linear

sensor_json_path = '/beegfs/sk7898/nodes.json'
feats_path = '/beegfs/work/sonyc/features/openl3/2017/' #Ex: sonycnode-b827eb132382.sonyc_features_openl3.h5

# Load the json file with all nodes' information
with open(sensor_json_path, 'r') as f:
    datastore = json.load(f)

sensor_feats_list = [s.split('_')[0] for s in os.listdir(feats_path)]
sensors_list = []
sensors_loc = []

# Get the latitude and longitude of the sensors from which l3 features were extracted given:
# 1. l3 feature file has size > 1432 (is not blank)
# 2. latitude and longitude values are present in the json file
for sensor_data in datastore:
    sensor = sensor_data['fqdn'] 
    if sensor in sensor_feats_list:
        feat_size = os.path.getsize(os.path.join(feats_path, sensor + '_features_openl3.h5'))
        if sensor_data['latitude'] and sensor_data['longitude'] and feat_size > 6585176:
            sensors_list.append(sensor)
            sensors_loc.append([radians(sensor_data['latitude']), radians(sensor_data['longitude'])])
        elif feat_size > 1432:
            print('Latitude and/or Longitude not present for {} of size {}'.format(sensor, feat_size))

L = haversine_distances(np.array(sensors_loc), np.array(sensors_loc))
L = 1/(L + 1e-15)

k = 14
DPP = FiniteDPP('likelihood', **{'L': L})
DPP.flush_samples()
DPP.sample_mcmc_k_dpp(size=k)
print(DPP.list_of_samples)

subset = DPP.list_of_samples[0][0] 

sampled_sensors = [sensors_list[i] for i in subset]
sampled_distances = [sensors_loc[i] for i in subset]
#print(sampled_sensors)

for sensor_data in datastore:
    sensor = sensor_data['fqdn'] 
    if sensor in sampled_sensors:
        print(sensor)
        #print('Sensor: {} Location: {}'.format(sensor, sensor_data['title']))
