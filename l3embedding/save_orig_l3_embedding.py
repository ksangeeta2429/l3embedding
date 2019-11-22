import os
import random
import math
import numpy as np
import h5py
import tempfile
import tensorflow as tf
from kapre.time_frequency import Melspectrogram
import keras
from keras.layers import Activation
from keras.models import Model
from skimage import img_as_float
from keras import activations
from l3embedding.audio import pcm2float
from keras import backend as K

# Do not allocate all the memory for visible GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

def apply_modifications(model, custom_objects=None):
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

def write_to_h5(path, batch):
    with h5py.File(path, 'a') as f:
        for key in batch.keys():
            if key in f.keys():
                continue
            f.create_dataset(key, data=batch[key], compression='gzip')
        f.close()


def get_teacher_logits(model, video_batch, audio_batch):
    if model is None:
        raise ValueError('Teacher L3 not provided!')
    
    test = False
    softmax = model.predict([video_batch, audio_batch])
    
    model.layers[-1].activation = activations.linear
    model = apply_modifications(model, custom_objects={'Melspectrogram': Melspectrogram})
    logits = model.layers[-1].output
    
    model_WO_softmax = Model(model.input, logits)
    predicted_logits = model_WO_softmax.predict([video_batch, audio_batch])

    if test:
        probabilities = Activation('softmax')(logits)
        verification_model = Model(model.input, probabilities)
        softmax_test = verification_model.predict([video_batch, audio_batch])
        print(softmax[:10])
        print(predicted_logits[:10])
        print(softmax_test[:10])
        exit(0)
 
    return predicted_logits, softmax

def freeze(model):
    """Freeze model weights in every layer."""
    for layer in model.layers:
        layer.trainable = False

        if isinstance(layer, keras.models.Model):
            freeze(layer)

def generate_output_driver(data_dir, output_dir, out_type='l3_embedding', partition_to_run=None,\
                           num_partitions=10, start_idx=None, **kwargs):
    #Divide l files in n-sized chunks
    def divide_chunks(l, n, start_idx=0):
        for i in range(start_idx, len(l), n):
            yield l[i:i+n]

    all_files = os.listdir(data_dir)
    
    if start_idx == None:
        idx = 0
        for fname in all_files:
            out_path = os.path.join(output_dir, fname)
            if os.path.exists(out_path) and out_type in h5py.File(out_path,'r').keys():
                idx += 1
                continue

        start_idx = idx + 1

    num_files = len(all_files)
    remaining_files = all_files[start_idx: num_files]
    all_files = list(divide_chunks(all_files, math.ceil(num_files / num_partitions), start_idx))

    # Get list of files to run
    print('Starting index: {}'.format(start_idx))
    print('Partition to run: {} out of {} partitions'.format(partition_to_run, num_partitions))
    list_files = all_files[partition_to_run]

    #print(list_files)
    #exit(0)
    embedding_generator(data_dir=data_dir, output_dir=output_dir, out_type=out_type, list_files=list_files, **kwargs)


def embedding_generator(data_dir, output_dir, out_type='l3_embedding', list_files=None, batch_size=256, **kwargs):
    
    if data_dir == output_dir:
        raise ValueError('Output path should not be same as data path to avoid overwriting data files!')
        
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if list_files == None:
        list_files = os.listdir(data_dir)

    weight_path = '/scratch/sk7898/l3pruning/embedding/fixed/reduced_input/l3_full_original_48000_256_242_2048.h5'
    model = keras.models.load_model(weight_path, custom_objects={'Melspectrogram': Melspectrogram}) 

    idx = 0

    for fname in list_files:
        blob_start_idx = 0
        out_blob = None
                
        out_path = os.path.join(output_dir, fname)
        if os.path.exists(out_path) and out_type in h5py.File(out_path,'r').keys():
            idx += 1
            continue

        batch_path = os.path.join(data_dir, fname)
        blob = h5py.File(batch_path, 'r')
        
        if out_type == 'embedding':
            batch = {'audio': blob['audio']}
            batch['audio'] = pcm2float(batch['audio'], dtype='float32')
            audio_model = model.get_layer('audio_model')
            out_blob['l3_embedding'] = audio_model.predict(batch['audio'])

        elif out_type == 'logits':
            batch = {'audio': blob['audio'], 'video': blob['video']}
            batch['audio'] = pcm2float(batch['audio'], dtype='float32')
            batch['video'] = 2 * img_as_float(batch['video']).astype('float32') - 1
            
            blob_size = len(batch['audio'])
            
            while blob_start_idx < blob_size:
                blob_end_idx = min(blob_start_idx + batch_size, blob_size)
                audio_batch = batch['audio'][blob_start_idx:blob_end_idx]
                video_batch = batch['video'][blob_start_idx:blob_end_idx]

                logits_out, softmax_out = get_teacher_logits(model, video_batch, audio_batch)
                if out_blob is None:
                    out_blob = {'logits': logits_out, 'softmax': softmax_out}
                else:
                    out_blob['logits'] = np.concatenate([out_blob['logits'], logits_out])
                    out_blob['softmax'] = np.concatenate([out_blob['softmax'], softmax_out])
                 
                blob_start_idx = blob_end_idx
        else:
            raise ValueError('Output type is not supported!')

        write_to_h5(out_path, out_blob) 
    
        idx += 1
        print('File {}: {} done!'.format(idx, fname))
        blob.close()
