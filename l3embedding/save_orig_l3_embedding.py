import os
import random
import numpy as np
import h5py
from l3embedding.audio import pcm2float
import tensorflow as tf
from kapre.time_frequency import Melspectrogram
import keras


def write_to_h5(path, batch):
    with h5py.File(path, 'a') as f:
        for key in batch.keys():
            if key in f.keys():
                continue
            f.create_dataset(key, data=batch[key], compression='gzip')
        f.close()


def embedding_generator(data_dir, output_dir, random_state=20180123):

    if data_dir == output_dir:
        raise ValueError('Output path should not be same as data path to avoid overwriting data files!')
        
    random.seed(random_state)
                
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    weight_path = '/scratch/sk7898/l3pruning/embedding/fixed/reduced_input/l3_full_original_48000_256_242_2048.h5'
    model = keras.models.load_model(weight_path, custom_objects={'Melspectrogram': Melspectrogram}) 
    audio_model = model.get_layer('audio_model')

    idx = 0
    for fname in os.listdir(data_dir):
        blob_embeddings = dict()
        
        batch_path = os.path.join(data_dir, fname)
        blob = h5py.File(batch_path, 'r')
        
        embedding_out_path = os.path.join(output_dir, fname)
        if os.path.exists(embedding_out_path):
            idx += 1
            print('Skipping file {}: {}! Already exists!'.format(idx, fname))
            continue

        batch = {'audio': blob['audio']}
                
        # Convert audio to float
        batch['audio'] = pcm2float(batch['audio'], dtype='float32')
                    
        # Get the embedding layer output from the audio_model and flatten it to be treated as labels for the student audio model
        blob_embeddings['l3_embedding'] = audio_model.predict(batch['audio'])
        write_to_h5(embedding_out_path, blob_embeddings) 
    
        idx += 1
        print('File {}: {} done!'.format(idx, fname))
        blob.close()
