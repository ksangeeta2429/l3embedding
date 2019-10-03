import csv
from multiprocessing import Pool
import glob
import h5py
import os
import math

def get_sonyc_filtered_files(csv_path):
    csvread = csv.reader(open(csv_path, newline=''), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    list_files = [row for row in csvread][0]
    return list_files


def check_sonyc_openl3_points(feature_dir, out_path, verbose=True,
                              min_num_datasets_per_file=0, max_num_datasets_per_file=math.inf):
    num_pts = 0
    num_sets = 0
    files = glob.glob(os.path.join(feature_dir, '*/*.h5'))
    valid_files=[]
    for fname in files:
        f = h5py.File(fname, 'r')
        if min_num_datasets_per_file < f[list(f.keys())[0]].shape[0] < max_num_datasets_per_file:
            valid_files += [fname]
            num_sets += f[list(f.keys())[0]].shape[0]
            if verbose:
                print('File: {} Num. of datasets: {}'.format(fname, f[list(f.keys())[0]].shape[0]))
            num_pts += f[list(f.keys())[0]].shape[0] * f[list(f.keys())[0]][0][1].shape[0]

    print('Num files:', len(valid_files))
    print('Num points:', num_pts)
    print('Num datasets:', num_sets)

    # Write valid files to csv
    csvwrite = csv.writer(open(out_path, 'w', newline=''), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwrite.writerow(valid_files)

def read_csv_as_dicts(path):
    items = []

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)

    return items

def map_iterate_in_parallel(iterable, function, processes=8):
    pool = Pool(processes=processes)
    output = pool.map(function, iterable)
    return list(output)


def flatten_dict(dct, parent_key=None):
    new_dct = type(dct)()
    for k,v in dct.items():
        if parent_key is not None:
            k = '{}_{}'.format(parent_key, k)

        if isinstance(v, dict):
            new_dct.update(flatten_dict(v, parent_key=k))
        else:
            new_dct[k] = v

    return new_dct

# if __name__=='__main__':
    # get_sonyc_filtered_files('/scratch/dr2915/reduced_embeddings/sonyc_files_list.csv')
    # check_sonyc_openl3_points('/beegfs/work/sonyc/features/openl3_day_format/2017',
    #                           '/scratch/dr2915/reduced_embeddings/sonyc_files_list.csv',
    #                           min_num_datasets_per_file=1400, max_num_datasets_per_file=1500)