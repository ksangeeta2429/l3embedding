import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import random
import shutil
import math
import numpy as np
import h5py
import tempfile
import tensorflow as tf
import multiprocessing
from kapre.time_frequency import Melspectrogram
from keras.layers import concatenate, Dense
import keras
from keras.layers import Activation
from keras.models import Model
from l3embedding.vision_model import *
from skimage import img_as_float
from keras import activations
from l3embedding.audio import pcm2float
from keras import backend as K
from functools import partial

# Do not allocate all the memory for visible GPU
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
K.set_session(sess)

def L3_merged_model(model, embedding_length=512, layer_size=128):
    
    # Merge the subnetworks
    weight_decay = 1e-5
    
    vision_model, x_i, y_i = construct_cnn_L3_orig_inputbn_vision_model()
    x_a = Input(shape=(embedding_length,), dtype='float32')
    
    y = concatenate([vision_model(x_i), x_a])
    y = Dense(layer_size, activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = Dense(2, activation='softmax',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(weight_decay))(y)
    m = Model(inputs=[x_i, x_a], outputs=y)
    
    m.get_layer('vision_model').set_weights(model.get_layer('vision_model').get_weights())
    m.layers[-1].set_weights(model.layers[-1].get_weights())
    m.layers[-2].set_weights(model.layers[-2].get_weights())

    return m, [x_i, x_a], y

def apply_temp_modifications(model, custom_objects=None):
    """Applies modifications to the model layers to create a new Graph. For example, simply changing
    `model.layers[idx].activation= new activation` does not change the graph. The entire graph needs to be updated
    with modified inbound and outbound tensors because of change in layer building function.
    Args:
        model: The `keras.models.Model` instance.
    Returns:
        The modified model with changes applied. Does not mutate the original `model`.
    """
    # The strategy is to save the modified model and load it back. This is done because setting the activation
    # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
    # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
    # multiple inbound and outbound nodes are allowed with the Graph API.
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return keras.models.load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)
        
def save_logits_model(model, out_path, custom_objects=None):
    model.save(out_path)
    return keras.models.load_model(out_path, custom_objects=custom_objects)

def get_teacher_logits(model, video_batch, audio_batch):
    if model is None:
        raise ValueError('Teacher L3 not provided!')

    logits = model.layers[-1].output
    
    model_WO_softmax = Model(model.input, logits)
    predicted_logits = model_WO_softmax.predict([video_batch, audio_batch])

    return predicted_logits

def get_teacher_softmax(model, video_batch, audio_batch):
    softmax = model.predict([video_batch, audio_batch])
    return softmax 

def write_to_h5(f, batch):
    for key in f.keys():
        if key in f.keys():
            continue
        f.create_dataset(key, data=batch[key], compression='gzip')
    f.close()
    
def generate_output_driver(data_dir, output_dir, out_type='l3_embedding', partition_to_run=None,\
                           num_partitions=10, start_idx=None, **kwargs):

    #Divide l files in n-sized chunks
    def divide_chunks(l, n, start_idx=0):
        for i in range(start_idx, len(l), n):
            yield l[i:i+n]

    all_files = os.listdir(data_dir)
    new_dir = None

    if 'valid' in output_dir:
        new_dir = '/scratch/sk7898/orig_l3_embeddings/music_valid_new'
    elif 'train' in output_dir:
        new_dir = '/scratch/sk7898/orig_l3_embeddings/music_train_logits'

    remaining_files = os.listdir(output_dir)
    all_files = list(divide_chunks(remaining_files, math.ceil(len(remaining_files) / num_partitions)))

    print('Partition to run: {} out of {} partitions'.format(partition_to_run, num_partitions))
    list_files = all_files[partition_to_run]

    embedding_generator(data_dir=data_dir, new_dir=new_dir, output_dir=output_dir, out_type=out_type, list_files=list_files, **kwargs)


def embedding_generator(data_dir, output_dir, new_dir=None, out_type='l3_embedding', list_files=None, batch_size=64, **kwargs):
    
    if data_dir == output_dir:
        raise ValueError('Output path should not be same as data path to avoid overwriting data files!')
        
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if list_files == None:
        list_files = os.listdir(data_dir)

    weight_path = '/scratch/sk7898/l3pruning/embedding/fixed/reduced_input/l3_full_original_48000_256_242_2048.h5'
    model = keras.models.load_model(weight_path, custom_objects={'Melspectrogram': Melspectrogram}) 
    
    if out_type == 'logits' or out_type == 'softmax':
        out_path = '/scratch/sk7898/l3pruning/embedding/fixed/reduced_input/l3_full_original_48000_256_242_2048_logits.h5'
        model, inputs, outputs = L3_merged_model(model)
        #print(model.summary()) 
        
        if os.path.exists(out_path):
            logits_model = keras.models.load_model(out_path)
        else:
            model.layers[-1].activation = activations.linear
            logits_model = save_logits_model(model, out_path, custom_objects={'Melspectrogram': Melspectrogram})

    idx = 0

    for fname in list_files:
        blob_start_idx = 0
        out_blob = None
                
        out_path = os.path.join(output_dir, fname)
        new_path = os.path.join(new_dir, fname)
        if os.path.exists(new_path):
            if out_type in h5py.File(new_path,'r').keys() and 'l3_embedding' in h5py.File(new_path,'r').keys():
                idx += 1
                continue
            elif os.path.exists(out_path) and 'l3_embedding' in h5py.File(out_path,'r').keys():
                os.remove(new_path)
            else:
                print('Some corruption in file:', new_path)
                continue

        batch_path = os.path.join(data_dir, fname)
        blob = h5py.File(batch_path, 'r')
        emb_blob = h5py.File(out_path, 'a')
        
        if out_type == 'embedding':
            batch = {'audio': blob['audio']}
            batch['audio'] = pcm2float(batch['audio'], dtype='float32')
            audio_model = model.get_layer('audio_model')
            out_blob['l3_embedding'] = audio_model.predict(batch['audio'])

        elif out_type == 'logits':
            batch = {'audio_emb': emb_blob['l3_embedding'], 'video': blob['video']}
            batch['video'] = 2 * img_as_float(batch['video']).astype('float32') - 1            
            blob_size = len(batch['video'])
            
            while blob_start_idx < blob_size:
                blob_end_idx = min(blob_start_idx + batch_size, blob_size)
                audio_batch = batch['audio_emb'][blob_start_idx:blob_end_idx]
                video_batch = batch['video'][blob_start_idx:blob_end_idx]

                logits_out = get_teacher_logits(logits_model, video_batch, audio_batch)
                if out_blob is None:
                    out_blob = {'logits': logits_out}
                else:
                    out_blob['logits'] = np.concatenate([out_blob['logits'], logits_out])

                blob_start_idx = blob_end_idx

        elif out_type == 'softmax':
            batch = {'audio_emb': emb_blob['l3_embedding'], 'video': blob['video']}
            batch['video'] = 2 * img_as_float(batch['video']).astype('float32') - 1
            
            blob_size = len(batch['video'])
            
            while blob_start_idx < blob_size:
                blob_end_idx = min(blob_start_idx + batch_size, blob_size)
                audio_batch = batch['audio_emb'][blob_start_idx:blob_end_idx]
                video_batch = batch['video'][blob_start_idx:blob_end_idx]

                softmax_out = get_teacher_softmax(model, video_batch, audio_batch)
                if out_blob is None:
                    out_blob = {'softmax': softmax_out}
                else:
                    out_blob['softmax'] = np.concatenate([out_blob['softmax'], softmax_out])
                    
        else:
            raise ValueError('Output type is not supported!')

        write_to_h5(emb_blob, out_blob)
        shutil.move(out_path, new_dir) 
    
        idx += 1
        print('File {}: {} done!'.format(idx, fname))
        blob.close()
        emb_blob.close()
