import csv
import glob
import h5py
import os
import math
import pescador
import multiprocessing
import time
import numpy as np
import random
from joblib import Parallel, delayed, dump, load

def downsample_sonyc_points(csv_path, output_dir, sample_size, random_state=20180123, max_workers=30, embeddings_per_file=1024):
    def divide_chunks(l, n):
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def process_partition(datasets, jobindex):
        @pescador.streamable
        def random_feature_generator(h5_path):
            f = h5py.File(h5_path, 'r')
            num_datasets = f[list(f.keys())[0]].shape[0]
            while True:
                dataset_index = np.random.randint(0, num_datasets)
                num_features = f[list(f.keys())[0]][dataset_index][1].shape[0]
                feature_index = np.random.randint(0, num_features)
                yield f[list(f.keys())[0]][dataset_index][1][feature_index]

        streams = [random_feature_generator(x) for x in datasets]
        rate = math.ceil(sample_size_per_job / len(streams))
        print(multiprocessing.current_process(),
              'Num. of pescador streams: {}; Rate: {}'.format(len(streams), rate))

        mux = pescador.StochasticMux(streams, n_active=20, rate=rate, mode='single_active')

        num_files_per_job = sample_size_per_job // embeddings_per_file

        accumulator = []
        splitindex = 1
        start_time = time.time()
        for data in mux(max_iter=sample_size_per_job):
            accumulator += [data]
            if len(accumulator)==embeddings_per_file:
                outfile = h5py.File(os.path.join(output_dir,'sonyc_ndata={}_job={}_split={}.h5'.format(sample_size, jobindex, splitindex)), 'w')
                outfile.create_dataset('l3_embedding', data=np.array(accumulator))
                end_time = time.time()
                print(multiprocessing.current_process(), 'Wrote {}/{} files, processing time: {} s'
                      .format(splitindex, num_files_per_job, (end_time - start_time)))
                accumulator = []
                splitindex += 1
                start_time = time.time()


    assert sample_size > 0

    random.seed(random_state)

    all_files = get_sonyc_filtered_files(csv_path)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    num_files = len(all_files)
    num_jobs = min(multiprocessing.cpu_count(), max_workers)

    print('Number of jobs: {}'.format(num_jobs))
    sample_size_per_job = sample_size // num_jobs
    print('Sample size per job: {}'.format(sample_size_per_job))

    # Split file list into chunks
    all_files = list(divide_chunks(all_files, math.ceil(num_files / num_jobs)))

    # Begin parallel jobs
    Parallel(n_jobs=num_jobs)(delayed(process_partition)(list_files, jobindex) for jobindex, list_files in enumerate(all_files, 1))


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


# if __name__=='__main__':
    # get_sonyc_filtered_files('/scratch/dr2915/reduced_embeddings/sonyc_files_list.csv')
    # check_sonyc_openl3_points('/beegfs/work/sonyc/features/openl3_day_format/2017',
    #                           '/scratch/dr2915/reduced_embeddings/sonyc_files_list.csv',
    #                           min_num_datasets_per_file=1400, max_num_datasets_per_file=1500)