import os
import random
import csv
import numpy as np
import keras
from keras.optimizers import Adam
import pescador
from skimage import img_as_float
from audio import pcm2float
import h5py
from model import MODELS, load_model

def cycle_shuffle(iterable, shuffle=True):
    lst = list(iterable)
    while True:
        yield from lst
        if shuffle:
            random.shuffle(lst)


def data_generator(data_dir, batch_size=512, random_state=20180123,
                   start_batch_idx=None, keys=None):
    random.seed(random_state)

    batch = None
    curr_batch_size = 0
    batch_idx = 0

    # Limit keys to avoid producing batches with all of the metadata fields
    if not keys:
        keys = ['audio', 'video', 'label']

    for fname in cycle_shuffle(os.listdir(data_dir)):
        batch_path = os.path.join(data_dir, fname)
        blob_start_idx = 0

        blob = h5py.File(batch_path, 'r')
        blob_size = len(blob['label'])

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
                    # Preprocess video so samples are in [-1,1]
                    batch['video'] = 2 * img_as_float(batch['video']).astype('float32') - 1

                    # Convert audio to float
                    batch['audio'] = pcm2float(batch['audio'], dtype='float32')

                    yield batch

                batch_idx += 1
                
                curr_batch_size = 0
                batch = None

def single_epoch_data_generator(data_dir, epoch_size, **kwargs):
    while True:
        data_gen = data_generator(data_dir, **kwargs)
        for idx, item in enumerate(data_gen):
            yield item
            # Once we generate all batches for an epoch, restart the generator
            if (idx + 1) == epoch_size:
                break

def get_labels(train_data_dir, validation_data_dir, validation_epoch_size=1024,train_epoch_size=4096,
               train_batch_size=64, learning_rate=1e-4, validation_batch_size=64,
               model_type='cnn_L3_melspec1', random_state=20180123, gpus=1):

    #weight_path_melspec2 = '/scratch/jtc440/sonyc-usc/embedding/music/cnn_L3_melspec2/20180223113902/model_best_valid_accuracy_1gpu.h5' 
    weight_path = '/home/sk7898/l3embedding/models/cnn_l3_melspec1/model_best_valid_accuracy_1gpu.h5'
    # '/scratch/jtc440/sonyc-usc/embedding/music/cnn_L3_melspec1/20180221105528/model_best_valid_accuracy_1gpu.h5'
    m, inputs, outputs  = load_model(weight_path, model_type, return_io=True, src_num_gpus=1)

    loss = 'binary_crossentropy'
    metrics = ['accuracy']

    m.compile(Adam(lr=learning_rate),
              loss=loss,
              metrics=metrics)


    train_gen = data_generator(
        train_data_dir,
        batch_size=train_batch_size,
        random_state=random_state,
        start_batch_idx=None)

    train_gen = pescador.maps.keras_tuples(train_gen,
                                           ['video', 'audio'],
                                           'label')

    val_gen = single_epoch_data_generator(
        validation_data_dir,
        validation_epoch_size,
        batch_size=validation_batch_size,
        random_state=random_state)

    val_gen = pescador.maps.keras_tuples(val_gen,
                                         ['video', 'audio'],
                                         'label')

    n_batches = int(np.floor((1024 * 58622)/train_batch_size))
    #n_batches = 937,952 for batch size = 64
    #n_batches = 117,244 for batch size = 512
    #print(n_batches)
    n_iter =  int(np.floor(n_batches/500))   

    train_probs = []
    for i in range(n_iter): 
        train_probs = m.predict_generator(train_gen, steps=500, verbose=1)
        csvfile = '/home/sk7898/l3embedding/l3embedding/train_labels/train_probs_'+str(i)+'.csv'
        with open(csvfile, "w") as output:
            writer = csv.writer(output, delimiter=',')
            writer.writerows(train_probs) 
            train_probs = []   

    if(len(train_probs) != 0):
        csvfile = '/home/sk7898/l3embedding/l3embedding/train_labels/train_probs_'+str(n_iter+1)+'.csv'
        with open(csvfile, "w") as output:
            writer = csv.writer(output, delimiter=',')
            writer.writerows(train_probs)

    #val_probs = m.predict_generator(val_gen, steps=validation_epoch_size)
    
    return train_probs #, val_probs

train_data_dir = '/beegfs/work/AudioSetSamples/music_train' #'/beegfs/work/AudioSetSamples_environmental/urban_train'
validation_data_dir = '/beegfs/work/AudioSetSamples/music_valid' # _environmental/urban_valid'
train_probabilities = get_labels(train_data_dir, validation_data_dir)

#csvfile = 'train_probs.csv'
#with open(csvfile, "w") as output:
#    writer = csv.writer(output, delimiter=',')
#    writer.writerows(train_probabilities)
    
