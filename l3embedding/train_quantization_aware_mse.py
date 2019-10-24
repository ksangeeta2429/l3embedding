import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import getpass
import git
import os
import random
import csv
import datetime
import json
import pickle
import numpy as np
import keras
import pescador
import tensorflow as tf
import h5py
import tempfile
import librosa
from keras import backend as K
from keras import activations
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from .training_utils import conv_dict_to_val_list, multi_gpu_model, \
    MultiGPUCheckpointCallback, LossHistory, GSheetLogger, TimeHistory
from .model import *
from .audio import pcm2float
from log import *
from kapre.time_frequency import Spectrogram, Melspectrogram
from resampy import resample

LOGGER = logging.getLogger('quantization_aware_mse')
LOGGER.setLevel(logging.DEBUG)

def cycle_shuffle(iterable, shuffle=True):
    lst = list(iterable)
    while True:
        yield from lst
        if shuffle:
            random.shuffle(lst)

def amplitude_to_db(S, amin=1e-10, dynamic_range=80.0):
    magnitude = np.abs(S)
    power = np.square(magnitude, out=magnitude)
    ref_value = power.max()

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= log_spec.max()

    log_spec = np.maximum(log_spec, -dynamic_range)
    return log_spec

def get_melspectrogram(frame, n_fft=2048, mel_hop_length=242, samp_rate=48000, n_mels=256, fmax=None):
    S = np.abs(librosa.core.stft(frame, n_fft=n_fft, hop_length=mel_hop_length, window='hann', center=True, pad_mode='constant'))
    S = librosa.feature.melspectrogram(sr=samp_rate, S=S, n_fft=n_fft, n_mels=n_mels, fmax=fmax, power=1.0, htk=True)
    S = amplitude_to_db(np.array(S))
    return S

def get_embedding_length(model):
    embed_layer = model.get_layer('audio_embedding_layer')
    emb_len = tuple(embed_layer.get_output_shape_at(0))
    return emb_len[-1]

def get_embedding_key(method, batch_size, emb_len, neighbors=None, \
                      metric=None, min_dist=None):
    
    if method == 'umap':
        if neighbors is None or metric is None or min_dist is None:
            raise ValueError('Either neighbors or metric or min_dist is missing')
        
        key = 'umap_batch_' + str(batch_size) +\
              '_len_' + str(emb_len) +\
              '_k_' + str(neighbors) +\
              '_metric_' + metric +\
              '_dist|iter_' + str(min_dist)

    return key

def data_generator(data_dir, emb_dir, student_emb_length=None, approx_mode='umap', approx_train_size=None,\
                   neighbors=10, min_dist=0.3, metric='euclidean', batch_size=512, \
                   n_fft=2048, n_mels=256, n_hop=242, hop_size=0.1, fmax=None,\
                   samp_rate=16000, random_state=20180123, start_batch_idx=None, test=False):

    if approx_mode != 'mse' and (student_emb_length is None or approx_train_size is None):
        raise ValueError('Either student embedding length or reduced embedding training size was not provided')

    random.seed(random_state)
    batch = None
    curr_batch_size = 0
    batch_idx = 0

    if student_emb_length != 512:
        emb_key = get_embedding_key(approx_mode, approx_train_size, student_emb_length, neighbors=neighbors, \
                                    metric=metric, min_dist=min_dist)
    else:
        emb_key = 'l3_embedding'
        
    if test:
        print('Testing phase')
        data_list = os.listdir(data_dir)
    else:
        data_list = cycle_shuffle(os.listdir(data_dir))
        
    for fname in data_list:
        data_batch_path = os.path.join(data_dir, fname)
        emb_batch_path = os.path.join(emb_dir, fname)

        blob_start_idx = 0

        data_blob = h5py.File(data_batch_path, 'r')
        emb_blob = h5py.File(emb_batch_path, 'r')

        blob_size = len(data_blob['audio'])

        while blob_start_idx < blob_size:
            blob_end_idx = min(blob_start_idx + batch_size - curr_batch_size, blob_size)

            # If we are starting from a particular batch, skip computing all of
            # the prior batches
            if start_batch_idx is None or batch_idx >= start_batch_idx:
                if batch is None:
                    batch = {'audio': data_blob['audio'][blob_start_idx:blob_end_idx],\
                             'label': emb_blob[emb_key][blob_start_idx:blob_end_idx]}
                else:
                    batch['audio'] = np.concatenate([batch['audio'], data_blob['audio'][blob_start_idx:blob_end_idx]])
                    batch['label'] = np.concatenate([batch['label'], emb_blob[emb_key][blob_start_idx:blob_end_idx]])

            curr_batch_size += blob_end_idx - blob_start_idx
            blob_start_idx = blob_end_idx

            if blob_end_idx == blob_size:
                data_blob.close()
                emb_blob.close()

            if curr_batch_size == batch_size:
                X = []
                # If we are starting from a particular batch, skip yielding all
                # of the prior batches
                if start_batch_idx is None or batch_idx >= start_batch_idx:
                    # Convert audio to float
                    if(samp_rate==48000):
                        batch['audio'] = pcm2float(batch['audio'], dtype='float32')
                    else:
                        batch['audio'] = resample(pcm2float(batch['audio'], dtype='float32'), sr_orig=48000,
                                                  sr_new=samp_rate)

                    X = [get_melspectrogram(batch['audio'][i].flatten(), n_fft=n_fft, mel_hop_length=n_hop,\
                                            samp_rate=samp_rate, n_mels=n_mels) for i in range(batch_size)]

                    batch['audio'] = np.array(X)[:, :, :, np.newaxis]
                    #print(np.shape(batch['audio'])) #(64, 256, 191, 1)
                    yield batch

                batch_idx += 1
                curr_batch_size = 0
                batch = None


def single_epoch_data_generator(data_dir, emb_dir, epoch_size, **kwargs):
    while True:
        data_gen = data_generator(data_dir, emb_dir, **kwargs)
        for idx, item in enumerate(data_gen):
            yield item
            # Once we generate all batches for an epoch, restart the generator
            if (idx + 1) == epoch_size:
                break

def initialize_uninitialized_variables(sess):
    if hasattr(tf, 'global_variables'):
        variables = tf.global_variables()
    else:
        variables = tf.all_variables()

    #print(variables)
    uninitialized_variables = []
    for v in variables:
        if not hasattr(v, '_keras_initialized') or not v._keras_initialized:
            uninitialized_variables.append(v)
            v._keras_initialized = True
    
    #print(uninitialized_variables)
    if uninitialized_variables:
        if hasattr(tf, 'variables_initializer'):
            sess.run(tf.variables_initializer(uninitialized_variables))
        else:
            sess.run(tf.initialize_variables(uninitialized_variables)) 

def load_l3_audio_model(model_path): 
    model = keras.models.load_model(model_path)
    embed_layer = model.get_layer('audio_embedding_layer')
    pool_size = tuple(embed_layer.get_output_shape_at(0)[1:3])
    y_a = MaxPooling2D(pool_size=pool_size, padding='same')(model.output)
    y_a = Flatten()(y_a)
    model = keras.models.Model(inputs=model.input, outputs=y_a)
    return model

def get_model_params(model_description):
    fmax = None
    splits = model_description.split('_') 
    samp_rate = int(splits[3])
    n_mels = int(splits[4])
    n_hop = int(splits[5])
    n_fft = int(splits[6])
    if len(splits) == 10:
        fmax = int(splits[-1])
    return samp_rate, n_mels, n_hop, n_fft, fmax

def train_quantized_model(train_data_dir, validation_data_dir, emb_train_dir, emb_valid_dir, output_dir, model_path=None,\
                          approx_mode='umap', approx_train_size=None, neighbors=10, min_dist=0.3, metric='euclidean', tsne_iter=300,\
                          num_epochs=300, train_epoch_size=4096, validation_epoch_size=1024, gpus=1, \
                          n_mels=256, n_hop=242, n_dft=2048, samp_rate=48000, fmax=None, halved_convs=False,\
                          train_batch_size=64, validation_batch_size=64, log_path=None, disable_logging=False, \
                          random_state=20180216,learning_rate=0.00001, verbose=True, checkpoint_interval=10,\
                          continue_model_dir=None, gsheet_id=None, google_dev_app_name=None,):
    
    K.clear_session()
    init_console_logger(LOGGER, verbose=verbose)
    if not disable_logging:
        init_file_logger(LOGGER, log_path=log_path)

    LOGGER.debug('Initialized logging.')
    LOGGER.info('Embedding Reduction Mode: %s', approx_mode)

    #reduced_emb_dir_train = os.path.join(reduced_emb_dir, os.path.basename(train_data_dir))
    #reduced_emb_dir_valid = os.path.join(reduced_emb_dir, os.path.basename(validation_data_dir))
      
    if approx_mode == 'umap':
        if min_dist is None:
            min_dist = 0.3
        if metric is None:
            metric='euclidean'
        model_attribute = 'quantized_umap_train_' + str(approx_train_size) + '_neighbors_' + str(neighbors)+\
                            '_dist_' + str(min_dist) + '_metric_' + metric
    elif approx_mode == 'mse':
        min_dist = 0
        neighbors = 0
        metric = ''
        model_attribute = 'quantized_mse_original'
    else:
        raise ValueError('Approximation mode: {} not supported!'.format(approx_mode))
    
    
    #train graph
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    train_graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    train_sess = tf.Session(config=config, graph=train_graph)
    K.set_session(train_sess)

    with train_graph.as_default():
        # Set up callbacks
        if continue_model_dir:
            model_desc = continue_model_dir.split('/')[-3]
            latest_model_path = os.path.join(continue_model_dir, 'model_latest.h5')
            student_base_model = keras.models.load_model(latest_model_path, custom_objects={'Melspectrogram': Melspectrogram})
            samp_rate, n_mels, n_hop, n_dft, fmax = get_model_params(model_desc)

        elif model_path:
            model_desc = os.path.basename(model_path).strip('.h5')
            student_base_model = load_l3_audio_model(model_path)
            samp_rate, n_mels, n_hop, n_dft, fmax = get_model_params(model_desc)
        else:
            raise ValueError('Both continue_dir and model_path are not provided!')
        
        if 'half' in model_desc:
            model_repr = str(samp_rate)+'_'+str(n_mels)+'_'+str(n_hop)+'_'+str(n_dft)+'_half'+'_fmax_'+str(fmax)
        else:
            model_repr = str(samp_rate)+'_'+str(n_mels)+'_'+str(n_hop)+'_'+str(n_dft)+'_fmax_'+str(fmax)

        # Make sure the directories we need exist
        if continue_model_dir:
            model_dir = continue_model_dir
        else:
            model_dir = os.path.join(output_dir, 'embedding_approx', model_repr, model_attribute,\
                                     datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        student_emb_len = get_embedding_length(student_base_model);
        LOGGER.info('Student sampling rate: {}'.format(samp_rate))
        LOGGER.info('Model Representation: {}'.format(model_repr))
        LOGGER.info('Model Attribute: {}'.format(model_attribute))
        LOGGER.info('Student Embedding Length: {}'.format(student_emb_len))
        LOGGER.info('Model files can be found in "{}"'.format(model_dir))

        param_dict = {
            'username': getpass.getuser(),
            'model_dir': model_dir,
            'train_data_dir': train_data_dir,
            'validation_data_dir': validation_data_dir,
            'reduced_emb_train_dir': emb_train_dir,
            'reduced_emb_valid_dir': emb_valid_dir,
            'approx_mode': approx_mode,
            'neighbors': neighbors,
            'min_dist': min_dist,
            'metric': metric,
            'student_model_repr': model_repr,
            'student_emb_len': student_emb_len, 
            'num_epochs': num_epochs,
            'train_epoch_size': train_epoch_size,
            'validation_epoch_size': validation_epoch_size,
            'train_batch_size': train_batch_size,
            'validation_batch_size': validation_batch_size,
            'random_state': random_state,
            'learning_rate': learning_rate,
            'verbose': verbose,
            'checkpoint_interval': checkpoint_interval,
            'log_path': log_path,
            'disable_logging': disable_logging,
            'gpus': gpus,
            'continue_model_dir': continue_model_dir,
        }

        LOGGER.info('Training with the following arguments: {}'.format(param_dict))

        param_dict.update({
            'latest_epoch': '-',
            'latest_train_loss': '-',
            'latest_validation_loss': '-',
            'latest_train_mae': '-',
            'latest_validation_mae': '-',
            'best_train_loss': '-',
            'best_validation_loss': '-',
            'best_train_mae': '-',
            'best_validation_mae': '-',
        })

        latest_weight_path = os.path.join(model_dir, 'model_latest.h5')
        best_valid_mae_weight_path = os.path.join(model_dir, 'model_best_valid_mae.h5')
        best_valid_loss_weight_path = os.path.join(model_dir, 'model_best_valid_loss.h5')
        checkpoint_weight_path = os.path.join(model_dir, 'model_checkpoint.{epoch:02d}.h5')

        # Load information about last epoch for initializing callbacks and data generators
        if continue_model_dir is not None:
            prev_train_hist_path = os.path.join(continue_model_dir, 'history_csvlog.csv')
            last_epoch_idx, last_val_mae, last_val_loss = get_restart_info(prev_train_hist_path)
        
        cb = []
        if gsheet_id:
            cb.append(GSheetLogger(google_dev_app_name, gsheet_id, param_dict))
            
        cb.append(MultiGPUCheckpointCallback(latest_weight_path,
                                             student_base_model,
                                             save_weights_only=False,
                                             verbose=1))

        best_val_mae_cb = MultiGPUCheckpointCallback(best_valid_mae_weight_path,
                                                     student_base_model,
                                                     save_weights_only=False,\
                                                     save_best_only=True,\
                                                     verbose=1,\
                                                     monitor='val_mean_absolute_error')
        if continue_model_dir is not None:
            best_val_mae_cb.best = last_val_mae
        cb.append(best_val_mae_cb)

        best_val_loss_cb = MultiGPUCheckpointCallback(best_valid_loss_weight_path,
                                                      student_base_model,
                                                      save_weights_only=False,
                                                      save_best_only=True,
                                                      verbose=1,
                                                      monitor='val_loss')
        if continue_model_dir is not None:
            best_val_loss_cb.best = last_val_loss
        cb.append(best_val_loss_cb)

        checkpoint_cb = MultiGPUCheckpointCallback(checkpoint_weight_path,
                                                   student_base_model,
                                                   save_weights_only=False,
                                                   period=checkpoint_interval)
        if continue_model_dir is not None:
            checkpoint_cb.epochs_since_last_save = (last_epoch_idx + 1) % checkpoint_interval
        cb.append(checkpoint_cb)

        history_checkpoint = os.path.join(model_dir, 'history_checkpoint.pkl')
        cb.append(LossHistory(history_checkpoint))

        history_csvlog = os.path.join(model_dir, 'history_csvlog.csv')
        cb.append(keras.callbacks.CSVLogger(history_csvlog, append=True, separator=','))

        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10)
        reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

        cb.append(earlyStopping)
        cb.append(reduceLR)
    
        LOGGER.info('Setting up train data generator...')
        if continue_model_dir is not None:
            train_start_batch_idx = train_epoch_size * (last_epoch_idx + 1)
        else:
            train_start_batch_idx = None
            
        train_gen = data_generator(train_data_dir,
                                   emb_train_dir,
                                   approx_mode=approx_mode,
                                   approx_train_size=approx_train_size,
                                   neighbors=neighbors,
                                   min_dist=min_dist,
                                   student_emb_length=student_emb_len,
                                   samp_rate=samp_rate,
                                   batch_size=train_batch_size,
                                   n_fft=n_dft, n_mels=n_mels, n_hop=n_hop, fmax=fmax,
                                   random_state=random_state,
                                   start_batch_idx=train_start_batch_idx)

        val_gen = single_epoch_data_generator(validation_data_dir,
                                              emb_valid_dir,
                                              validation_epoch_size,
                                              approx_mode=approx_mode,
                                              approx_train_size=approx_train_size,
                                              neighbors=neighbors,
                                              min_dist=min_dist,
                                              student_emb_length=student_emb_len,
                                              samp_rate=samp_rate,
                                              batch_size=validation_batch_size,
                                              n_fft=n_dft, n_mels=n_mels, n_hop=n_hop, fmax=fmax,
                                              random_state=random_state)


        train_gen = pescador.maps.keras_tuples(train_gen,
                                               'audio',
                                               'label')

        val_gen = pescador.maps.keras_tuples(val_gen,
                                             'audio',
                                             'label')
    
        # Fit the model
        LOGGER.info('Fitting model...')
        if verbose:
            verbosity = 1
        else:
            verbosity = 2

        if continue_model_dir is not None:
            initial_epoch = last_epoch_idx + 1
        else:
            initial_epoch = 0
            
        optimizer = Adam(lr=learning_rate)
        #print(tf.all_variables())
        #print(tf.get_default_graph().get_operations())
        tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
        initialize_uninitialized_variables(train_sess)
        
        #Convert the base (single-GPU) model to Multi-GPU model
        if gpus == 1:
            model = student_base_model
        else:
            model = multi_gpu_model(student_base_model, gpus=gpus)
    
        model.compile(optimizer,\
                      loss='mean_squared_error',\
                      metrics=['mae'])
        
        history = model.fit_generator(train_gen, train_epoch_size, num_epochs,\
                                      validation_data=val_gen,
                                      validation_steps=validation_epoch_size,
                                      callbacks=cb,
                                      verbose=verbosity, initial_epoch=initial_epoch)

        #save graph and checkpoints
        saver = tf.train.Saver()
        saver.save(train_sess, model_dir) 
    
        LOGGER.info('Done training. Saving results to disk...')
        # Save history
        with open(os.path.join(model_dir, 'history.pkl'), 'wb') as fd:
            pickle.dump(history.history, fd)

    LOGGER.info('Done!')