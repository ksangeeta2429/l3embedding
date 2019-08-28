import os
import random
import numpy as np
import h5py
from l3embedding.audio import pcm2float
import umap
from sklearn.manifold import TSNE
import multiprocessing
from joblib import Parallel, delayed


def get_teacher_embedding(audio_batch):
    import keras
    import tensorflow as tf
    from kapre.time_frequency import Melspectrogram
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    try:
        with tf.Graph().as_default(), tf.Session().as_default(), tf.device('/gpu:0'):
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


def get_reduced_embedding(data, method, emb_len=None, neighbors=10, metric='euclidean', \
                min_dist=0.3, iterations=500):
    
    if len(data) == 0:
        raise ValueError('Data is empty!')
    if emb_len is None:
        raise ValueError('Reduced embedding dimension was not provided!')

    if method == 'umap':
        embedding = umap.umap_.UMAP(n_neighbors=neighbors, min_dist=min_dist, metric=metric, \
                                    n_components=emb_len).fit_transform(data)
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

def embedding_generator(data_dir, output_dir, reduced_emb_len, approx_mode='umap', neighbors_list=None, \
                        metric_list=None, min_dist_list=None, tsne_iter_list=[500], \
                        batch_size=1024, random_state=20180123, start_batch_idx=None):

    if data_dir == output_dir:
        raise ValueError('Output path should not be same as data path to avoid overwriting data files!')
        
    if neighbors_list is None:
        raise ValueError('Neighbor cannot be None!')
    
    if metric_list is None:
        metric_list = ['euclidean']

    if approx_mode == 'umap' and min_dist_list is None:
        min_dist_list = [0.3]
    
    random.seed(random_state)
    
    batch = None
    embedding_out_paths = []
    curr_batch_size = 0
    batch_idx = 0

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    for fname in os.listdir(data_dir):
        print('Data filename: {}'.format(fname))

        batch_path = os.path.join(data_dir, fname)
        blob_start_idx = 0

        blob = h5py.File(batch_path, 'r')
        blob_size = len(blob['label'])
        
        embedding_out_paths.append(os.path.join(output_dir, fname))

        while blob_start_idx < blob_size:
            blob_end_idx = min(blob_start_idx + batch_size - curr_batch_size, blob_size)

            # If we are starting from a particular batch, skip computing all of
            # the prior batches
            if start_batch_idx is None or batch_idx >= start_batch_idx:
                if batch is None:
                    batch = {'audio':blob['audio'][blob_start_idx:blob_end_idx]}
                else:
                    batch['audio'] = np.concatenate([batch['audio'], blob['audio'][blob_start_idx:blob_end_idx]])

            curr_batch_size += blob_end_idx - blob_start_idx
            blob_start_idx = blob_end_idx
            
            if blob_end_idx == blob_size:
                blob.close()

            if curr_batch_size == batch_size:
                blob_embeddings = dict()
                blob_keys = get_blob_keys(approx_mode, batch_size, reduced_emb_len, \
                                          neighbors_list=neighbors_list, metric_list=metric_list, \
                                          min_dist_list=min_dist_list, tsne_iter_list=tsne_iter_list)
    
                print('Embedding Blob Keys: {}'.format(blob_keys))
        
                # If we are starting from a particular batch, skip yielding all
                # of the prior batches
                if start_batch_idx is None or batch_idx >= start_batch_idx:
                    # Convert audio to float
                    batch['audio'] = pcm2float(batch['audio'], dtype='float32')
                    
                    # Get the embedding layer output from the audio_model and flatten it to be treated as labels for the student audio model
                    teacher_embedding = get_teacher_embedding(batch['audio'])
                    
                    if approx_mode == 'umap':
                        n_process = len(neighbors_list) * len(metric_list) * len(min_dist_list)
                        
                        results = Parallel(n_jobs=-1)(delayed(get_reduced_embedding)(teacher_embedding, 'umap', \
                                                                                     emb_len=reduced_emb_len, \
                                                                                     neighbors=neighbors, \
                                                                                     metric=metric, \
                                                                                     min_dist=min_dist) \
                                          for neighbors in neighbors_list for metric in metric_list for min_dist in min_dist_list)

                    elif approx_mode == 'tsne':
                        n_process = len(neighbors_list) * len(metric_list) * len(tsne_iter_list)
                        
                        results = Parallel(n_jobs=-1)(delayed(get_reduced_embedding)(teacher_embedding, 'tsne', \
                                                                                     emb_len=reduced_emb_len, \
                                                                                     neighbors=neighbors, \
                                                                                     metric=metric, \
                                                                                     iterations=iterations) \
                                          for neighbors in neighbors_list for metric in metric_list for iterations in tsne_iter_list)

                        assert len(results) == n_process
                        
                    for idx in range(len(results)):
                        if blob_keys[idx] not in blob_embeddings.keys():    
                            blob_embeddings[blob_keys[idx]] = np.zeros((blob_size, reduced_emb_len), dtype=np.float32)
                            blob_embeddings[blob_keys[idx]] = results[idx]
                        else:
                            blob_embeddings[blob_keys[idx]] = results[idx]
                      
                        blob_embeddings['l3_embedding'] = teacher_embedding
                        write_to_h5(embedding_out_paths, blob_embeddings, batch_size) 
                        print('Batch completed!')
                
                batch_idx += 1
                curr_batch_size = 0
                batch = None 
                embedding_out_paths = []
