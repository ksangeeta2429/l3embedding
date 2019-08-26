import os
import random
import numpy as np
import keras
import tensorflow as tf
import h5py
from .model import *
from .audio import pcm2float
import umap
from sklearn.manifold import TSNE

graph = tf.get_default_graph()
weight_path = 'models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5'
audio_model = load_embedding(weight_path, model_type = 'cnn_L3_melspec2', embedding_type = 'audio', \
                             pooling_type = 'short', kd_model=False, tgt_num_gpus = 1)


def get_embedding(data, method, emb_length=None, neighbors=10, min_dist=0.3, iterations=300):
    if len(data) == 0:
        raise ValueError('Data is empty!')
    if emb_length is None:
        raise ValueError('Reduced embedding dimension was not provided!')

    if method == 'umap':
        embedding = umap.umap_.UMAP(n_neighbors=neighbors, min_dist=min_dist, \
                                    n_components=emb_length).fit_transform(data)
    elif method == 'tsne':
        embedding = TSNE(perplexity=neighbors, n_components=emb_length, n_iter=iterations).fit_transform(data)
    else:
        raise ValueError('Reduction method technique should be either `umap` or `tsne`!')
    
    return embedding

def cycle_shuffle(iterable, shuffle=True):
    lst = list(iterable)
    while True:
        yield from lst
        if shuffle:
            random.shuffle(lst)

def data_generator(data_dir, reduced_emb_len, output_dir, reduction_method='umap', neighbors=10, min_dist=0.3, tsne_iter=300, \
                   batch_size=1024, random_state=20180123, start_batch_idx=None):
    
    random.seed(random_state)

    batch = None
    global graph
    global audio_model
    curr_batch_size = 0
    batch_idx = 0
    keys = ['audio']

    for fname in cycle_shuffle(os.listdir(data_dir)):
        batch_path = os.path.join(data_dir, fname)
        blob_start_idx = 0

        blob = h5py.File(batch_path, 'r')
        blob_size = len(blob['label'])
        embeddings = np.zeros((blob_size, reduced_emb_len), dtype=np.float32)
        if data_dir == output_dir:
            raise ValueError('Output path should not be same as Data path to avoid overwriting data files!')
        embedding_out_path = os.path.join(output_dir, fname)

        if reduction_method == 'umap':
            embedding_key = reduction_method + '_batch_' + str(batch_size) + '_k_' + str(neighbors) + '_dist_' + str(min_dist)
        elif reduction_method == 'tsne':
            embedding_key = reduction_method + '_batch_' + str(batch_size) + '_k_' + str(neighbors) + '_iter_' + str(tsne_iter)

        while blob_start_idx < blob_size:
            blob_end_idx = min(blob_start_idx + batch_size - curr_batch_size, blob_size)

            # If we are starting from a particular batch, skip computing all of
            # the prior batches
            if start_batch_idx is None or batch_idx >= start_batch_idx:
                if batch is None:
                    batch = {k:blob[k][blob_start_idx:blob_end_idx]
                             for k in keys}
                else:
                    for k in keys:
                        batch[k] = np.concatenate([batch[k],
                                                   blob[k][blob_start_idx:blob_end_idx]])

            curr_batch_size += blob_end_idx - blob_start_idx
            blob_start_idx = blob_end_idx

            if blob_end_idx == blob_size:
                blob.close()

            if curr_batch_size == batch_size:
                # If we are starting from a particular batch, skip yielding all
                # of the prior batches
                if start_batch_idx is None or batch_idx >= start_batch_idx:
                    # Convert audio to float
                    batch['audio'] = pcm2float(batch['audio'], dtype='float32')
                    # Get the embedding layer output from the audio_model and flatten it to be treated as labels for the student audio model
                    with graph.as_default():
                        teacher_embedding = audio_model.predict(batch['audio'])

                    embeddings[blob_start_idx:blob_end_idx,:] = get_embedding(teacher_embedding, reduction_method, neighbors=neighbors, \
                                                                              min_dist=min_dist, iterations=tsne_iter)

                batch_idx += 1
                curr_batch_size = 0
                batch = None

        with h5py.File(embedding_out_path, 'a') as f:
            f.create_dataset(embedding_key, data=embeddings)
            f.close()


