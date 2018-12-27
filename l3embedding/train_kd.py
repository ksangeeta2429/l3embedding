import os
import random
import csv
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


def load_student_audio_model_preFFT():
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
    m.name = 'student_audio_model'

    return m, x_a, y_a


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
        data_gen = data_generator(data_dir, soft_labels_dir, **kwargs)
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
                    batch['audio'] = pcm2float(batch['audio'], dtype='float32')
                        
                    # Get the embedding layer output from the audio_model and flatten it to be treated as labels for the student audio model
                    with graph.as_default():
                        batch['label'] = audio_model.predict(batch['audio'])
                                            
                    yield batch

                batch_idx += 1
                curr_batch_size = 0
                batch = None


def train(train_data_dir, validation_data_dir, weight_path, model_type = 'cnn_L3_melspec2', num_epochs=300, validation_epoch_size=1024,
          train_epoch_size=4096, train_batch_size=64, learning_rate=1e-4, validation_batch_size=64,
          random_state=20180216, gpus=1):

    #m, inputs, outputs = load_model(weight_path, model_type, return_io=True, src_num_gpus=1)
    #audio_model = m.get_layer('audio_model')
    
    train_gen = data_generator(
        train_data_dir,
        batch_size=train_batch_size,
        random_state=random_state,
        start_batch_idx=None)

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

    student, x_a, y_a = load_student_audio_model()

    optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)

    student.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mae', 'acc'])

    history = student.fit_generator(train_gen, train_epoch_size, num_epochs,
                                    validation_data=val_gen,
                                    validation_steps=validation_epoch_size,
                                    verbose=1)

    return history

train_data_dir = '/beegfs/work/AudioSetSamples/music_train' # _environmental/urban_train'
validation_data_dir = '/beegfs/work/AudioSetSamples/music_valid' # _environmental/urban_valid'
history = train(train_data_dir, validation_data_dir, weight_path)
