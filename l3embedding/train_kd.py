import os
import random
import csv
import datetime
import json
import pickle
import numpy as np
from keras.regularizers import l2
import tensorflow as tf
import keras
from keras.optimizers import Adam
import pescador
from keras.layers import *
from audio import pcm2float
import h5py
from keras.models import Model
from model import *
from keras.optimizers import Adam
import pescador
from skimage import img_as_float
from keras import backend as K

graph = tf.get_default_graph()
weight_path = '/home/sk7898/l3embedding/models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5'
audio_model = load_embedding(weight_path, model_type = 'cnn_L3_melspec2', embedding_type = 'audio', pooling_type = 'short', tgt_num_gpus = 1)


##########
# 1. Added student model : load_student_audio_model
# 2. data_generator has audio_model passed for generating the embedding output which acts as label for student model
# 3. The student is compiled with MSE loss with metric as mae 
#    (MSE is chosen for loss because of many nan issues that I have seen during training but RMSE can also be tried)
##########

def cycle_shuffle(iterable, shuffle=True):
    lst = list(iterable)
    while True:
        yield from lst
        if shuffle:
            random.shuffle(lst)


def load_student_audio_model():
    weight_decay = 1e-5
    ####
    # Audio subnetwork
    ####
    n_dft = 2048
    #n_win = 480
    #n_hop = n_win//2
    n_mels = 256
    n_hop = 242
    asr = 48000
    audio_window_dur = 1
    # INPUT

    n_frames = 1 + int((asr * audio_window_dur) / float(n_hop))
    x_a = Input(shape=(n_mels, n_frames, 1), dtype='float32')
    y_a = BatchNormalization()(x_a)

    # CONV BLOCK 1
    n_filter_a_1 = 64
    filt_size_a_1 = (3, 3)
    pool_size_a_1 = (2, 2)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    #y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
    #             kernel_initializer='he_normal',
    #             kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    #y_a = BatchNormalization()(y_a)
    #y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_1, strides=2)(y_a)

    # CONV BLOCK 2
    n_filter_a_2 = 128
    filt_size_a_2 = (3, 3)
    pool_size_a_2 = (2, 2)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    ###
    #y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
    #             kernel_initializer='he_normal',
    #             kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    #y_a = BatchNormalization()(y_a)
    #y_a = Activation('relu')(y_a)
    ###
    y_a = MaxPooling2D(pool_size=pool_size_a_2, strides=2)(y_a)

    # CONV BLOCK 3
    n_filter_a_3 = 256
    filt_size_a_3 = (3, 3)
    pool_size_a_3 = (2, 2)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    ###
    #y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
    #             kernel_initializer='he_normal',
    #             kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    #y_a = BatchNormalization()(y_a)
    #y_a = Activation('relu')(y_a)
    ###
    y_a = MaxPooling2D(pool_size=pool_size_a_3, strides=2)(y_a)

    # CONV BLOCK 4
    n_filter_a_4 = 512
    filt_size_a_4 = (3, 3)
    pool_size_a_4 = (32, 24)
    #y_a = Conv2D(n_filter_a_4, filt_size_a_4, padding='same',
    #             kernel_initializer='he_normal',
    #             kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    #y_a = BatchNormalization()(y_a)
    #y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4,
                 kernel_initializer='he_normal',
                 name='student_embedding_layer', padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_4)(y_a)

    y_a = Flatten()(y_a)

    m = Model(inputs=x_a, outputs=y_a)
    m.name = 'student_audio_model'

    return m, x_a, y_a


def load_student_audio_model_withFFT():
    weight_decay = 1e-5
    ####
    # Audio subnetwork
    ####
    n_dft = 2048
    #n_win = 480
    #n_hop = n_win//2
    n_mels = 256
    n_hop = 242
    asr = 48000
    audio_window_dur = 1
    # INPUT
    x_a = Input(shape=(1, asr * audio_window_dur), dtype='float32')

    # MELSPECTROGRAM PREPROCESSING
    # 128 x 199 x 1
    y_a = Melspectrogram(n_dft=n_dft, n_hop=n_hop, n_mels=n_mels,
                      sr=asr, power_melgram=1.0, htk=True, # n_win=n_win,
                      return_decibel_melgram=True, padding='same')(x_a)
    y_a = BatchNormalization()(y_a)

    # CONV BLOCK 1
    n_filter_a_1 = 64
    filt_size_a_1 = (3, 3)
    pool_size_a_1 = (2, 2)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_1, strides=2)(y_a)

    # CONV BLOCK 2
    n_filter_a_2 = 128
    filt_size_a_2 = (3, 3)
    pool_size_a_2 = (2, 2)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    ###
    #y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
    #             kernel_initializer='he_normal',
    #             kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    #y_a = BatchNormalization()(y_a)
    #y_a = Activation('relu')(y_a)
    ###
    y_a = MaxPooling2D(pool_size=pool_size_a_2, strides=2)(y_a)

    # CONV BLOCK 3
    n_filter_a_3 = 256
    filt_size_a_3 = (3, 3)
    pool_size_a_3 = (2, 2)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    ###
    #y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
    #             kernel_initializer='he_normal',
    #             kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    #y_a = BatchNormalization()(y_a)
    #y_a = Activation('relu')(y_a)
    ###
    y_a = MaxPooling2D(pool_size=pool_size_a_3, strides=2)(y_a)

    # CONV BLOCK 4
    n_filter_a_4 = 512
    filt_size_a_4 = (3, 3)
    pool_size_a_4 = (32, 24)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4,
                 kernel_initializer='he_normal',
                 name='student_embedding_layer', padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_4)(y_a)

    y_a = Flatten()(y_a)

    m = Model(inputs=x_a, outputs=y_a)
    m.name = 'student_model'

    return m, x_a, y_a


def single_epoch_data_generator(data_dir, epoch_size, **kwargs):
    while True:
        data_gen = data_generator(data_dir, **kwargs)
        for idx, item in enumerate(data_gen):
            yield item
            # Once we generate all batches for an epoch, restart the generator
            if (idx + 1) == epoch_size:
                break


def data_generator(data_dir, batch_size=512, random_state=20180216, start_batch_idx=None, keys=None):
    random.seed(random_state)

    batch = None
    curr_batch_size = 0
    batch_idx = 0
    file_idx = 0
    start_label_idx = 0
    global graph
    global audio_model

    # Limit keys to avoid producing batches with all of the metadata fields
    if not keys:
        keys = ['audio']

    for fname in cycle_shuffle(os.listdir(data_dir)):
        batch_path = os.path.join(data_dir, fname)

        blob_start_idx = 0

        blob = h5py.File(batch_path, 'r')
        blob_size = len(blob['label'])

        #embedding_layer_model = Model(inputs=audio_model.get_input_at(0), outputs=audio_model.get_layer('audio_embedding_layer').output)
        #embedding_layer_model._make_predict_function()

        while blob_start_idx < blob_size:
            #embedding_output = None
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
                    #batch['video'] = 2 * img_as_float(batch['video']).astype('float32') - 1

                    # Convert audio to float
                    audio_data_batch = batch['audio']
                    batch['audio'] = []

                    for audio_data in audio_data_batch:
                        audio_data = pcm2float(audio_data.flatten(), dtype='float32')
                        # Compute spectrogram
                        if model_type == 'cnn_L3_melspec2':
                            S = np.abs(minispec.core.stft(audio_data, n_fft=2048, hop_length=242,
                                                          window='hann', center=True,
                                                          pad_mode='constant'))
                            audio_data = minispec.feature.melspectrogram(sr=48000, S=S,
                                                                         n_mels=256, power=1.0,
                                                                         htk=True)
                            del S
                        elif model_type == 'cnn_L3_melspec1':
                            S = np.abs(minispec.core.stft(audio_data, n_fft=2048, hop_length=242,
                                                          window='hann', center=True,
                                                          pad_mode='constant'))
                            audio_data = minispec.feature.melspectrogram(sr=48000, S=S,
                                                                         n_mels=128, power=1.0,
                                                                         htk=True)
                            del S
                        else:

                            audio_data = np.abs(minispec.core.stft(audio_data, n_fft=512, hop_length=242,
                                                                   window='hann', center=True,
                                                                   pad_mode='constant'))

                        # Convert amplitude to dB
                        audio_data = minispec.core.amplitude_to_db(audio_data)

                        # Add additional batch and channel dimensions
                        batch['audio'].append(audio_data[np.newaxis,:,:,np.newaxis])

                        del audio_data

                    del audio_data_batch

                    #batch['audio'] = pcm2float(batch['audio'], dtype='float32')
                        
                    # Get the embedding layer output from the audio_model and flatten it to be treated as labels for the student audio model
                    with graph.as_default():
                        batch['label'] = audio_model.predict(batch['audio'])
                                            
                    yield batch

                batch_idx += 1
                curr_batch_size = 0
                batch = None


def train(train_data_dir, validation_data_dir, weight_path, output_dir = '/scratch/sk7898/kd_output', num_epochs=300, train_epoch_size=4096, 
          validation_epoch_size=1024, train_batch_size=64, validation_batch_size=64, model_type = 'cnn_L3_melspec2', random_state=20180216, 
          learning_rate=0.00001, verbose=True, checkpoint_interval=10, gpus=1):

    #m, inputs, outputs = load_model(weight_path, model_type, return_io=True, src_num_gpus=1)
    #audio_model = m.get_layer('audio_model')
    
    # Form model ID
    data_subset_name = os.path.basename(train_data_dir)
    data_subset_name = data_subset_name[:data_subset_name.rindex('_')]
    model_id = os.path.join(data_subset_name, model_type)
    model_dir = os.path.join(output_dir, 'embedding', model_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    param_dict = {
          'train_data_dir': train_data_dir,
          'validation_data_dir': validation_data_dir,
          'model_id': model_id,
          'output_dir': output_dir,
          'num_epochs': num_epochs,
          'train_epoch_size': train_epoch_size,
          'validation_epoch_size': validation_epoch_size,
          'train_batch_size': train_batch_size,
          'validation_batch_size': validation_batch_size,
          'model_type': model_type,
          'random_state': random_state,
          'learning_rate': learning_rate,
          'verbose': verbose
    }
    
    param_dict['model_dir'] = model_dir
    train_config_path = os.path.join(model_dir, 'config.json')
    with open(train_config_path, 'w') as fd:
        json.dump(param_dict, fd, indent=2)

    param_dict.update({
          'latest_epoch': '-',
          'latest_train_loss': '-',
          'latest_validation_loss': '-',
          'latest_train_acc': '-',
          'latest_validation_acc': '-',
          'best_train_loss': '-',
          'best_validation_loss': '-',
          'best_train_acc': '-',
          'best_validation_acc': '-',
    })

    student, x_a, y_a = load_student_audio_model()

    optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)

    student.compile(Adam(lr=learning_rate),
                    loss='mean_squared_error',
                    metrics=['mae'])

    # Save the model
    model_spec_path = os.path.join(model_dir, 'model_spec.pkl')
    model_spec = keras.utils.serialize_keras_object(student)
    with open(model_spec_path, 'wb') as fd:
        pickle.dump(model_spec, fd)
    model_json_path = os.path.join(model_dir, 'model.json')
    model_json = student.to_json()
    with open(model_json_path, 'w') as fd:
        json.dump(model_json, fd, indent=2)

    latest_weight_path = os.path.join(model_dir, 'model_latest.h5')
    best_valid_loss_weight_path = os.path.join(model_dir, 'model_best_valid_loss.h5')
    checkpoint_weight_path = os.path.join(model_dir, 'model_checkpoint.{epoch:02d}.h5')

    # Set up callbacks
    cb = []

    best_val_loss_cb = keras.callbacks.ModelCheckpoint(best_valid_loss_weight_path,
                                              save_weights_only=True,
                                              save_best_only=True,
                                              verbose=1,
                                              monitor='val_loss')
    #if continue_model_dir is not None:
    #    best_val_loss_cb.best = last_val_loss
    cb.append(best_val_loss_cb)

    checkpoint_cb = keras.callbacks.ModelCheckpoint(checkpoint_weight_path,
                                              save_weights_only=True,
                                              period=checkpoint_interval)
    #if continue_model_dir is not None:
    #    checkpoint_cb.epochs_since_last_save = (last_epoch_idx + 1) % checkpoint_interval
    cb.append(checkpoint_cb)

    #LOGGER.info('Setting up train data generator...')
    #if continue_model_dir is not None:
    #    train_start_batch_idx = train_epoch_size * (last_epoch_idx + 1)
    #else:
    train_start_batch_idx = None

    train_gen = data_generator(
        train_data_dir,
        batch_size=train_batch_size,
        random_state=random_state,
        start_batch_idx=train_start_batch_idx)

    train_gen = pescador.maps.keras_tuples(train_gen,
                                           'audio',
                                           'label')

    val_gen = single_epoch_data_generator(
        validation_data_dir,
        validation_epoch_size,
        batch_size=validation_batch_size,
        random_state=random_state)

    val_gen = pescador.maps.keras_tuples(val_gen,
                                         'audio',
                                         'label')

    history = student.fit_generator(train_gen, train_epoch_size, num_epochs,
                                    validation_data=val_gen,
                                    validation_steps=validation_epoch_size,
                                    callbacks=cb,
                                    verbose=1)

    # Save history
    with open(os.path.join(model_dir, 'history.pkl'), 'wb') as fd:
        pickle.dump(history.history, fd)

    return history

train_data_dir = '/beegfs/work/AudioSetSamples/music_train' # _environmental/urban_train'
validation_data_dir = '/beegfs/work/AudioSetSamples/music_valid' # _environmental/urban_valid'
history = train(train_data_dir, validation_data_dir, weight_path)
