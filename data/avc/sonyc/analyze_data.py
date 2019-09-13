import os
import h5py
import glob
import numpy as np

path_audio = '/beegfs/work/sonyc/audio/2017'
path_indices = '/beegfs/work/sonyc/new_indices/2017'

list_sensors = os.listdir(path_audio)

full_dict = dict()

for sensor in list_sensors:
    os.chdir(os.path.join(path_audio, sensor))
    list_files = glob.glob('*.h5')

    for filename in list_files:
        f = h5py.File(filename, 'r')
        num_points = f[list(f.keys())[0]].shape[0]
        # print("File: {}\t#points: {}".format(os.path.join(os.getcwd(), filename), num_points))
        full_dict[os.path.abspath(filename)] = num_points

num_points_all = np.fromiter(full_dict.values(), dtype=int).sum()

print('########################################')
print('Total number of training points: {}'.format(num_points_all))
print('########################################')

os.chdir(path_indices)
list_indices = glob.glob('*.h5')

filtered_dict = dict()

for file in list_indices:
    ind = h5py.File(file, 'r')
    for k in ind[list(ind.keys())[0]]:
        audio_file = os.path.join('/beegfs/work/sonyc',k[1].decode("utf-8"))
        if audio_file not in filtered_dict.keys():
            num_points = full_dict[audio_file]
            print("File: {}\t#points: {}".format(os.path.join(os.getcwd(), audio_file), num_points))
            filtered_dict[audio_file] = num_points

num_points_filtered = np.fromiter(filtered_dict.values(), dtype=int).sum()

print('----------------------------------------')
print('Number of filtered training points: {}'.format(num_points_filtered))
print('----------------------------------------')