import os
import math
import random
import numpy as np
import pescador
import h5py
import umap
from sklearn.manifold import TSNE
import time
import multiprocessing
from joblib import Parallel, delayed, dump, load
import re
import glob
from sonyc.utils import get_sonyc_filtered_files

def get_teacher_embedding(audio_batch):
    import tensorflow as tf
    from kapre.time_frequency import Melspectrogram
    from l3embedding.model import load_embedding 

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import keras

    session_conf = tf.ConfigProto(
        device_count={'CPU' : 1, 'GPU' : 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    try:
        with tf.Graph().as_default(), tf.Session(config=session_conf).as_default():
            weight_path = '/scratch/sk7898/l3pruning/embedding/fixed/reduced_input/l3_full_original_48000_256_242_2048.h5'
            model = keras.models.load_model(weight_path, custom_objects={'Melspectrogram': Melspectrogram}) 
            embeddings = model.get_layer('audio_model').predict(audio_batch)
            return embeddings

    except GeneratorExit:
        pass


def write_to_h5(paths, batch, batch_size):
    n_files = int(batch_size/1024)
    start_idx = 0
    
    for path in paths:
        end_idx = start_idx + 1024
            
        with h5py.File(path, 'a') as f:
            for key in batch.keys():
                if key in f.keys():
                    continue
                f.create_dataset(key, data=batch[key][start_idx:end_idx], compression='gzip')
            f.close()
        start_idx = end_idx


# Note: For UMAP, if a saved model is provided, the UMAP params are all ignored
def get_reduced_embedding(data, method, emb_len=None, umap_estimator= None, neighbors=10, metric='euclidean', \
                          min_dist=0.3, iterations=500):
    
    if len(data) == 0:
        raise ValueError('Data is empty!')
    if emb_len is None:
        raise ValueError('Reduced embedding dimension was not provided!')

    if method == 'umap':
        if umap_estimator is None:
            embedding = umap.umap_.UMAP(n_neighbors=neighbors, min_dist=min_dist, metric=metric, \
                                        n_components=emb_len).fit_transform(data)
        else:
            start_time = time.time()
            embedding = umap_estimator.transform(data)
            end_time = time.time()

            print('UMAP extraction time for 1 batch: {} seconds'.format((end_time - start_time)))
    elif method == 'tsne':
        embedding = TSNE(perplexity=neighbors, n_components=emb_len, metric=metric, \
                         n_iter=iterations, method='exact').fit_transform(data)
    else:
        raise ValueError('Reduction method technique should be either `umap` or `tsne`!')
    
    return embedding


def get_blob_keys(method, batch_size, emb_len, neighbors_list=None, metric_list=None, min_dist_list=None, tsne_iter_list=None):
    
    blob_keys = []
    
    if method == 'umap':
        if neighbors_list is None or metric_list is None or min_dist_list is None:
            raise ValueError('Either neighbor_list or metric_list or min_dist_list is missing')
        
        [blob_keys.append('umap_batch_' + str(batch_size) + \
                          '_len_' + str(emb_len) + \
                          '_k_' + str(neighbors) + \
                          '_metric_' + metric + \
                          '_dist|iter_' + str(min_dist)) \
         for neighbors in neighbors_list for metric in metric_list for min_dist in min_dist_list]
                    
    elif method == 'tsne':
        if neighbors_list is None or metric_list is None or tsne_iter_list is None:
            raise ValueError('Either neighbor_list or metric_list or tsne_iter_list is missing')
        
        [blob_keys.append('tsne_batch_' + str(batch_size) +\
                          '_len_' + str(emb_len) + \
                          '_batch_' + str(batch_size) + \
                          '_k_' + str(neighbors) + \
                          '_metric_' + metric + \
                          '_dist|iter_' + str(iteration)) \
        for neighbors in neighbors_list for metric in metric_list for iteration in tsne_iter_list]

    return blob_keys


def embedding_generator(data_dir, output_dir, reduced_emb_len, approx_mode='umap', umap_estimator_path=None, neighbors_list=None, \
                        list_files = None, metric_list=None, min_dist_list=None, tsne_iter_list=[500], \
                        batch_size=1024, random_state=20180123, start_batch_idx=None):

    if data_dir == output_dir:
        raise ValueError('Output path should not be same as data path to avoid overwriting data files!')

    if neighbors_list is None and umap_estimator_path is None:
        raise ValueError('Neighbor cannot be None!')

    if metric_list is None and umap_estimator_path is None:
        metric_list = ['euclidean']

    if approx_mode == 'umap' and min_dist_list is None and umap_estimator_path is None:
        min_dist_list = [0.3]
    
    random.seed(random_state)
    
    batch = None
    blob_embeddings = dict()
    embedding_out_paths = []
    curr_batch_size = 0
    batch_idx = 0

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Infer UMAP params if path provided
    if umap_estimator_path is not None:
        # Infer training params from filename
        m = re.match('umap_ndata=(?P<_0>.+)_emb=(?P<_1>.+)_nbrs=(?P<_2>.+)_mindist=(?P<_3>.+)_mtrc=(?P<_4>.+)\.sav',
                     os.path.basename(umap_estimator_path))
        inferred_params = [y[1] for y in sorted(m.groupdict().items())]
        blob_keys = get_blob_keys('umap', int(inferred_params[0]), int(inferred_params[1]),
                                  neighbors_list=[int(inferred_params[2])], metric_list=[inferred_params[4]],
                                  min_dist_list=[float(inferred_params[3])], tsne_iter_list=tsne_iter_list)
        
        # Extract reducer
        print('Loading UMAP model...')
        start_time = time.time()
        reducer=load(umap_estimator_path)
        end_time = time.time()

        print('UMAP model loading: {} seconds'.format((end_time - start_time)))
    else:
        blob_keys = get_blob_keys(approx_mode, batch_size, reduced_emb_len, \
                                  neighbors_list=neighbors_list, metric_list=metric_list, \
                                  min_dist_list=min_dist_list, tsne_iter_list=tsne_iter_list)
    
    print('Embedding Blob Keys: {}'.format(blob_keys))

    # If a list of files is not provided, use all files in data_dir
    if list_files==None:
        list_files = os.listdir(data_dir)
    last_file = list_files[-1]
    print('Last file on the list: ', last_file)

    f_idx = 0
    for fname in list_files:
        batch_path = os.path.join(data_dir, fname)
        blob_start_idx = 0

        blob = h5py.File(batch_path, 'r')
        blob_size = len(blob['l3_embedding'])        

        embedding_out_paths.append(os.path.join(output_dir, fname))

        read_start = time.time()
        while blob_start_idx < blob_size:
            blob_end_idx = min(blob_start_idx + batch_size - curr_batch_size, blob_size)

            # If we are starting from a particular batch, skip computing all of
            # the prior batches
            if start_batch_idx is None or batch_idx >= start_batch_idx:
                if batch is None:
                    batch = {'l3_embedding':blob['l3_embedding'][blob_start_idx:blob_end_idx]} 
                else:
                    batch['l3_embedding'] = np.concatenate([batch['l3_embedding'], blob['l3_embedding'][blob_start_idx:blob_end_idx]])

            curr_batch_size += blob_end_idx - blob_start_idx
            blob_start_idx = blob_end_idx
            
            if blob_end_idx == blob_size:
                blob.close()

            if curr_batch_size == batch_size or (fname == last_file and blob_end_idx == blob_size):
                read_end = time.time()
                print('Batch reading: {} seconds'.format((read_end - read_start)))
                # If we are starting from a particular batch, skip yielding all
                # of the prior batches
                if start_batch_idx is None or batch_idx >= start_batch_idx:

                    teacher_embedding = batch['l3_embedding'] #get_teacher_embedding(batch['audio'])
                    
                    if approx_mode == 'umap':
                        if umap_estimator_path is None:
                            n_process = len(neighbors_list) * len(metric_list) * len(min_dist_list)
                            results = Parallel(n_jobs=min(multiprocessing.cpu_count(), n_process))\
                                      (delayed(get_reduced_embedding)(teacher_embedding, 'umap', \
                                                                      emb_len=reduced_emb_len, umap_estimator=None, \
                                                                      neighbors=neighbors, \
                                                                      metric=metric, \
                                                                      min_dist=min_dist) \
                                              for neighbors in neighbors_list for metric in metric_list for min_dist in min_dist_list)
                        else:
                            print('Batch size:', batch_size)
                            results = get_reduced_embedding(teacher_embedding, 'umap', emb_len=reduced_emb_len,
                                                            umap_estimator=reducer)
                    elif approx_mode == 'tsne':
                        n_process = len(neighbors_list) * len(metric_list) * len(tsne_iter_list)
                        
                        results = Parallel(n_jobs=n_process)(delayed(get_reduced_embedding)\
                                                             (teacher_embedding, 'tsne', \
                                                              emb_len=reduced_emb_len, \
                                                              neighbors=neighbors, \
                                                              metric=metric, \
                                                              iterations=iterations) \
                                          for neighbors in neighbors_list for metric in metric_list for iterations in tsne_iter_list)

                    if umap_estimator_path is None:
                        assert len(results) == n_process
                        
                        for idx in range(len(results)):
                            if blob_keys[idx] not in blob_embeddings.keys():
                                blob_embeddings[blob_keys[idx]] = np.zeros((batch_size, reduced_emb_len), dtype=np.float32)
                                blob_embeddings[blob_keys[idx]] = results[idx]
                            else:
                                blob_embeddings[blob_keys[idx]] = results[idx]
                    else:
                        if blob_keys[0] not in blob_embeddings.keys():
                            blob_embeddings[blob_keys[0]] = np.zeros((batch_size, reduced_emb_len), dtype=np.float32)
                            blob_embeddings[blob_keys[0]] = results
                        else:
                            blob_embeddings[blob_keys[0]] = results

                    write_start = time.time()
                    write_to_h5(embedding_out_paths, blob_embeddings, batch_size)
                    write_end = time.time()
                    f_idx += 1
                    print('File {}: {} done! Write took {} seconds'.format(f_idx, fname, (write_end-write_start)))
                    print('-----------------------------------------\n')

                batch_idx += 1
                curr_batch_size = 0
                batch = None 
                blob_embeddings = dict()
                embedding_out_paths = []
                read_start = time.time()


def generate_trained_umap_embeddings_driver(data_dir, output_dir, continue_extraction=False, partition_to_run=None, num_partitions=10, **kwargs):
    def divide_chunks(l, n):
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    if partition_to_run is None:
        # Compute remaining list of files
        if continue_extraction:
            list_files = list(set(os.listdir(data_dir))-set(os.listdir(output_dir)))
            print('Continuing to extract for {} remaining files'.format(len(list_files)))
        else:
            list_files = None
    else:
        # Run specified partition out of given num_partitions
        all_files = sorted(os.listdir(data_dir))
        num_files = len(all_files)

        # Split file list into chunks
        all_files = list(divide_chunks(all_files, math.ceil(num_files / num_partitions)))

        # Get list of files to run
        print('Partition to run: {} out of {} partitions'.format(partition_to_run, num_partitions))
        list_files = all_files[partition_to_run]

    # Call embedding generator with appropriate list of files
    embedding_generator(data_dir=data_dir, output_dir=output_dir, list_files=list_files, **kwargs)


def create_umap_training_dataset(data_dir, output_dir, training_size, random_state=20180123):
    def divide_chunks(l, n):
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    if 'sonyc' in data_dir:
        def process_partition(datasets, training_size_per_job, outfilename):
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
            rate = math.ceil(training_size_per_job / len(streams))
            print(multiprocessing.current_process(),
                  'Num. of pescador streams: {}; Rate: {}'.format(len(streams), rate))

            mux = pescador.StochasticMux(streams, n_active=20, rate=rate, mode='single_active')

            accumulator = []
            start_time = time.time()
            for data in mux(max_iter=training_size_per_job):
                accumulator += [data]
            outfile = h5py.File(outfilename, 'w')
            outfile.create_dataset('l3_embedding', data=np.array(accumulator))
            end_time = time.time()

            print(multiprocessing.current_process(), 'Wrote {}, processing time: {} s'
                  .format(outfilename, (end_time-start_time)))
    elif 'music' in data_dir:
        def process_partition(list_files, training_size_per_job, outfilename):
            # Find number of points in first pass
            num_points = 0
            for fname in list_files:
                f = h5py.File(fname, 'r')
                num_points += len(f['l3_embedding'])

            print(multiprocessing.current_process(), 'Number of files in chunk', len(list_files))
            print(multiprocessing.current_process(), 'Total number of points in chunk', num_points)
            print(multiprocessing.current_process(), 'Training size in chunk', training_size_per_job)
            print(multiprocessing.current_process(), 'Output file', outfilename)

            assert num_points >= training_size_per_job

            # Generate random numbers of training size
            indices = random.sample(range(num_points), training_size_per_job)
            indices.sort()
            training_pts = []
            running_index = 0

            next_sampled_index, indices = indices[0], indices[1:]
            for fname in list_files:
                batch_path = os.path.join(data_dir, fname)

                blob = h5py.File(batch_path, 'r')
                blob_size = len(blob['l3_embedding'])

                cur_file_start_index = running_index
                next_file_start_index = cur_file_start_index + blob_size

                # Populate training point(s)
                while running_index <= next_sampled_index < next_file_start_index:
                    # Add training point to list
                    training_pts.append(blob['l3_embedding'][next_sampled_index - cur_file_start_index])

                    # Update current index
                    running_index = next_sampled_index + 1

                    # Pop next index to sample
                    try:
                        next_sampled_index, indices = indices[0], indices[1:]
                        #print('Next index to be sampled', next_sampled_index)
                    except IndexError:
                        print(multiprocessing.current_process(), 'Creating UMAP training dataset')
                        outfile = h5py.File(outfilename, 'w')
                        outfile.create_dataset('l3_embedding', data=np.array(training_pts))
                        # Save dataset and exit when all indices have been sampled
                        blob.close()
                        return

                # Close blob
                running_index = next_file_start_index
                blob.close()

    if data_dir == output_dir:
        raise ValueError('Output path should not be same as data path to avoid overwriting data files!')

    assert training_size > 0

    random.seed(random_state)

    if 'music' in data_dir:
        os.chdir(data_dir)
        all_files = glob.glob('*.h5')
    elif 'sonyc' in data_dir:
        all_files = get_sonyc_filtered_files(data_dir)
    else:
        raise NotImplementedError('Invalid data directory: {}; can only be music or sonyc'.format(data_dir))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    num_files = len(all_files)
    num_jobs = min(multiprocessing.cpu_count(), 20)

    print('Number of jobs: {}'.format(num_jobs))
    training_size_per_chunk = training_size // num_jobs
    print('Training size per chunk: {}'.format(training_size_per_chunk))

    # Split file list into chunks
    all_files = list(divide_chunks(all_files, math.ceil(num_files/num_jobs)))

    # Begin parallel jobs
    Parallel(n_jobs=num_jobs)(delayed(process_partition)
                                                     (list_files,
                                                      training_size_per_chunk,
                                                      os.path.join(output_dir, 'umap_training_ndata={}_{}.h5'
                                                                   .format(training_size, index)))
                                                     for index, list_files in enumerate(all_files, 1))


def train_umap_embedding(data_dir, output_dir, reduced_emb_len, neighbors=5,
                    metric='euclidean', min_dist=0.3, batch_size=1024, random_state=20180123, start_batch_idx=None):
    if data_dir == output_dir:
        raise ValueError('Output path should not be same as data path to avoid overwriting data files!')

    random.seed(random_state)

    out_file_name = os.path.join(output_dir,
                                 'umap_ndata={}_emb={}_nbrs={}_mindist={}_mtrc={}.sav'.
                                 format(batch_size, reduced_emb_len, neighbors, min_dist, metric))

    print('Data directory:', data_dir)
    print('Model out filename:', out_file_name)

    batch = None
    curr_batch_size = 0
    batch_idx = 0

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for fname in os.listdir(data_dir):
        batch_path = os.path.join(data_dir, fname)
        blob_start_idx = 0

        blob = h5py.File(batch_path, 'r')
        blob_size = len(blob['l3_embedding'])

        while blob_start_idx < blob_size:
            blob_end_idx = min(blob_start_idx + batch_size - curr_batch_size, blob_size)

            # If we are starting from a particular batch, skip computing all of
            # the prior batches
            if start_batch_idx is None or batch_idx >= start_batch_idx:
                if batch is None:
                    batch = {'l3_embedding': blob['l3_embedding'][blob_start_idx:blob_end_idx]}
                else:
                    batch['l3_embedding'] = np.concatenate(
                        [batch['l3_embedding'], blob['l3_embedding'][blob_start_idx:blob_end_idx]])

            curr_batch_size += blob_end_idx - blob_start_idx
            blob_start_idx = blob_end_idx

            if blob_end_idx == blob_size:
                blob.close()

            # Use only the first full batch for training
            if curr_batch_size == batch_size:
                # If we are starting from a particular batch, skip yielding all
                # of the prior batches
                if start_batch_idx is None or batch_idx >= start_batch_idx:
                    teacher_embedding = batch['l3_embedding']  # get_teacher_embedding(batch['audio'])
                    print('Collected data; shape: {}'.format(teacher_embedding.shape))
                    np.random.shuffle(teacher_embedding)
                    print('Size in mem: {} GB'.format(teacher_embedding.nbytes/1e9))
                    reducer = umap.UMAP(n_neighbors=neighbors, min_dist=min_dist,
                                        metric=metric,n_components=reduced_emb_len, verbose=True)

                    print('Starting UMAP training: sample_size={}, num_neighbors={},'
                          'min_dist={}, metric={}, reduced_emb_len={}'.format(curr_batch_size, neighbors,
                                                                              min_dist, metric, reduced_emb_len))

                    start_time = time.time()
                    embedding = reducer.fit_transform(teacher_embedding)
                    end_time = time.time()

                    print('UMAP training finished: took {} hours'.format((end_time-start_time)/3600))

                    # Diagnostic
                    print('Train embedding shape: ', embedding.shape)

                    # Save pickled model
                    dump(reducer, out_file_name)
                    print('UMAP model saved at ', out_file_name)

                    return