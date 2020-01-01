import csv
import glob
import h5py
import warnings
import os
import math
import pescador
import multiprocessing
import time
import numpy as np
import random
import pickle
from joblib import Parallel, delayed, dump, load
from decrypt import read_encrypted_tar_audio_file

def create_dict_audio_tar_to_h5(audio_h5_path, out_dir):
    list_files = glob.glob(os.path.join(audio_h5_path, '*/*.h5'))
    map = dict()
    for file in list_files:
        f=h5py.File(file)
        for row in range(len(f['recordings'])):
            map[f['recordings'][row]['filename']] = (file, row)

    # Dump dictionary in pickle
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'map.pkl')) as f:
        pickle.dump(map, f)


def downsample_sonyc_points(feature_dir, audio_dir, output_dir, sample_size, audio_samp_rate=8000,
                            random_state=20180123, max_workers=30,
                            embeddings_per_file=1024):
    def divide_chunks(l, n):
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    # Streamer weights are proportional to the number of datasets in the corresponding file
    def generate_pescador_stream_weights(list_files):
        num_datasets = []
        for file in list_files:
            f = h5py.File(file)
            num_datasets.append(f[list(f.keys())[0]].shape[0])

        num_datasets = np.array(num_datasets)
        return num_datasets.astype(float) / num_datasets.sum()

    def process_partition(list_files, jobindex):
        @pescador.streamable
        def random_feature_generator(h5_path):
            f = h5py.File(h5_path, 'r')
            num_datasets = f[list(f.keys())[0]].shape[0]
            while True:
                dataset_index = np.random.randint(0, num_datasets)
                num_features = f[list(f.keys())[0]][dataset_index]['openl3'].shape[0]
                raw_audio = get_raw_windows_from_encrypted_audio(
                    os.path.join(audio_dir, f[list(f.keys())[0]][dataset_index]['filename']))
                feature_index = np.random.randint(0, num_features)
                yield f[list(f.keys())[0]][dataset_index]['openl3'][feature_index], raw_audio[feature_index]

        streams = [random_feature_generator(x) for x in list_files]
        rate = math.ceil(sample_size_per_job / len(streams))
        print(multiprocessing.current_process(),
              'Num. of pescador streams: {}; Rate: {}'.format(len(streams), rate))

        mux = pescador.StochasticMux(streams, weights=generate_pescador_stream_weights(list_files), n_active=20,
                                     rate=rate, mode='exhaustive')

        num_files_per_job = sample_size_per_job // embeddings_per_file

        accumulator = []
        rawlist = []
        splitindex = 1
        start_time = time.time()
        for data, raw in mux(max_iter=sample_size_per_job):
            accumulator += [data]
            rawlist += raw
            if len(accumulator) == embeddings_per_file:
                outfile = h5py.File(os.path.join(output_dir,
                                                 'sonyc_ndata={}_job={}_split={}.h5'.format(sample_size, jobindex,
                                                                                            splitindex)), 'w')
                outfile.create_dataset('audio', data=np.array(rawlist))
                outfile.create_dataset('l3_embedding', data=np.array(accumulator))
                end_time = time.time()
                print(multiprocessing.current_process(), 'Wrote {}/{} files, processing time: {} s'
                      .format(splitindex, num_files_per_job, (end_time - start_time)))
                accumulator = []
                splitindex += 1
                start_time = time.time()

    assert sample_size > 0

    random.seed(random_state)

    all_files = glob.glob(os.path.join(feature_dir, '*/*.h5'))

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
    Parallel(n_jobs=num_jobs)(
        delayed(process_partition)(list_files, jobindex) for jobindex, list_files in enumerate(all_files, 1))


def get_raw_windows_from_encrypted_audio(audio_path, sample_rate=8000, clip_duration=10,
                                         decrypt_url='https://decrypt-sonyc.engineering.nyu.edu/decrypt',
                                         cacert_path='/home/jtc440/sonyc/decrypt/CA.pem',
                                         cert_path='/home/jtc440/sonyc/decrypt/jason_data.pem',
                                         key_path='/home/jtc440/sonyc/decrypt/sonyc_key.pem'):
    audio = read_encrypted_tar_audio_file(audio_path.decode(),
                                          sample_rate=sample_rate,
                                          url=decrypt_url,
                                          cacert=cacert_path,
                                          cert=cert_path,
                                          key=key_path)[0]

    audio_len = int(sample_rate * clip_duration)

    # Make sure audio is all consistent length (10 seconds)
    if len(audio) > audio_len:
        audio = audio[:audio_len]
    elif len(audio) < audio_len:
        pad_len = audio_len - len(audio)
        audio = np.pad(audio, (0, pad_len), mode='constant')

    # Return raw windows
    return get_audio_windows(audio, sr=sample_rate)


def get_audio_windows(audio, sr=8000, center=True, hop_size=0.1):
    """
    Similar to openl3.get_embedding(...)
    """

    def _center_audio(audio, frame_len):
        """Center audio so that first sample will occur in the middle of the first frame"""
        return np.pad(audio, (int(frame_len / 2.0), 0), mode='constant', constant_values=0)

    def _pad_audio(audio, frame_len, hop_len):
        """Pad audio if necessary so that all samples are processed"""
        audio_len = audio.size
        if audio_len < frame_len:
            pad_length = frame_len - audio_len
        else:
            pad_length = int(np.ceil((audio_len - frame_len) / float(hop_len))) * hop_len \
                         - (audio_len - frame_len)

        if pad_length > 0:
            audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)

        return audio

    # Check audio array dimension
    if audio.ndim > 2:
        raise AssertionError('Audio array can only be be 1D or 2D')
    elif audio.ndim == 2:
        # Downmix if multichannel
        audio = np.mean(audio, axis=1)

    audio_len = audio.size
    frame_len = sr
    hop_len = int(hop_size * sr)

    if audio_len < frame_len:
        warnings.warn('Duration of provided audio is shorter than window size (1 second). Audio will be padded.')

    if center:
        # Center audio
        audio = _center_audio(audio, frame_len)

    # Pad if necessary to ensure that we process all samples
    audio = _pad_audio(audio, frame_len, hop_len)

    # Split audio into frames, copied from librosa.util.frame
    n_frames = 1 + int((len(audio) - frame_len) / float(hop_len))
    x = np.lib.stride_tricks.as_strided(audio, shape=(frame_len, n_frames),
                                        strides=(audio.itemsize, hop_len * audio.itemsize)).T

    # Add a channel dimension
    # x = x.reshape((x.shape[0], 1, x.shape[-1]))

    return x


def get_sonyc_filtered_files(csv_path):
    csvread = csv.reader(open(csv_path, newline=''), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    list_files = [row for row in csvread][0]
    return list_files


def check_sonyc_openl3_points(feature_dir, out_path, verbose=True,
                              min_num_datasets_per_file=0, max_num_datasets_per_file=math.inf):
    num_pts = 0
    num_sets = 0
    files = glob.glob(os.path.join(feature_dir, '*/*.h5'))
    valid_files = []
    for fname in files:
        f = h5py.File(fname, 'r')
        if min_num_datasets_per_file < f[list(f.keys())[0]].shape[0] < max_num_datasets_per_file:
            valid_files += [fname]
            num_sets += f[list(f.keys())[0]].shape[0]
            if verbose:
                print('File: {} Num. of datasets: {}'.format(fname, f[list(f.keys())[0]].shape[0]))
            num_pts += f[list(f.keys())[0]].shape[0] * f[list(f.keys())[0]][0]['openl3'].shape[0]

    print('Num files:', len(valid_files))
    print('Num points:', num_pts)
    print('Num datasets:', num_sets)

    # Write valid files to csv
    csvwrite = csv.writer(open(out_path, 'w', newline=''), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwrite.writerow(valid_files)

if __name__=='__main__':
    create_dict_audio_tar_to_h5('/beegfs/work/sonyc/audio/2017/', 'scratch/dr2915/sonyc_map')
# get_sonyc_filtered_files('/scratch/dr2915/reduced_embeddings/sonyc_files_list.csv')
# check_sonyc_openl3_points('/beegfs/work/sonyc/features/openl3_day_format/2017',
#                           '/scratch/dr2915/reduced_embeddings/sonyc_files_list.csv',
#                           min_num_datasets_per_file=1400, max_num_datasets_per_file=1500)
