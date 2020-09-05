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
import sys

# Do not allocate all the memory for visible GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)


LOGGER = logging.getLogger('embedding_approx_mse')
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

def get_student_model(model_path):
    l3model = keras.models.load_model(model_path)
    embed_layer = l3model.get_layer('audio_embedding_layer')
    pool_size = tuple(embed_layer.get_output_shape_at(0)[1:3])
    y_a = keras.layers.MaxPooling2D(pool_size=pool_size, padding='same')(l3model.output)
    y_a = keras.layers.Flatten()(y_a)
    model = keras.models.Model(inputs=l3model.input, outputs=y_a)
   
    return model

def get_model_params(model_description, continue_train=False):
    fmax = None
    splits = model_description.split('_')

    if continue_train:
        samp_rate = int(splits[0])
        n_mels = int(splits[1])
        n_hop = int(splits[2])
        n_fft = int(splits[3])
    else:
        samp_rate = int(splits[3])
        n_mels = int(splits[4])
        n_hop = int(splits[5])
        n_fft = int(splits[6])
    
    if 'fmax' in model_description and splits[-1] != "None":
        fmax = int(splits[-1])
    
    return samp_rate, n_mels, n_hop, n_fft, fmax
        
def get_embedding_length(model):
    embed_layer = model.get_layer('audio_embedding_layer')
    emb_len = tuple(embed_layer.get_output_shape_at(0))
    return emb_len[-1]

def get_melspectrogram(frame, n_fft=2048, mel_hop_length=242, samp_rate=48000, n_mels=256, fmax=None):
    S = np.abs(librosa.core.stft(frame, n_fft=n_fft, hop_length=mel_hop_length, window='hann', center=True, pad_mode='constant'))
    S = librosa.feature.melspectrogram(sr=samp_rate, S=S, n_fft=n_fft, n_mels=n_mels, fmax=fmax, power=1.0, htk=True)
    S = amplitude_to_db(np.array(S))
    return S

def get_embedding_key(method, batch_size, emb_len, neighbors=None, \
                      metric=None, min_dist=None, pca_kernel=None):
    
    if method == 'umap':
        if neighbors is None or metric is None or min_dist is None:
            raise ValueError('Either neighbors or metric or min_dist is missing')
        
        key = 'umap_batch_' + str(batch_size) +\
              '_len_' + str(emb_len) +\
              '_k_' + str(neighbors) +\
              '_metric_' + metric +\
              '_dist|iter_' + str(min_dist)
         
    elif method == 'pca':
        if pca_kernel is None:
            raise ValueError('PCA kernel is missing')

        blob_keys.append('pca_batch_' + str(batch_size) + \
                         '_len_' + str(emb_len) + \
                         '_kernel_' + str(pca_kernel))

    else:
        raise ValueError('Only UMAP and PCA is supported as of now!')

    return key

def get_restart_info(history_path):
    #epoch,loss,mean_absolute_error,val_loss,val_mean_absolute_error
    last = None
    with open(history_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row

    return int(last['epoch']), float(last['val_mean_absolute_error']), float(last['val_loss'])

def data_generator(data_dir, emb_dir, emb_key, batch_size=512, melSpec=False,
                   student_asr=8000, n_fft=2048, n_mels=256, n_hop=242, hop_size=0.1, fmax=None,
                   random_state=20180216, start_batch_idx=None):

    random.seed(random_state)
    batch = None
    curr_batch_size = 0
    batch_idx = 0

    if type(emb_dir) == list:
        file_list = emb_dir
    else:
        file_list = os.listdir(emb_dir)
        
    for fname in cycle_shuffle(file_list):
        data_batch_path = os.path.join(data_dir, fname)
        emb_batch_path = os.path.join(emb_dir, fname)

        blob_start_idx = 0

        # If file is unreadable for any reason, move on to the next file
        try:
            data_blob = h5py.File(data_batch_path, 'r')
            emb_blob = h5py.File(emb_batch_path, 'r')
        except:
            print("Unexpected error:", sys.exc_info()[1])
            continue

        blob_size = len(emb_blob[emb_key])

        while blob_start_idx < blob_size:
            #embedding_output = None
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
                # If we are starting from a particular batch, skip yielding all
                # of the prior batches
                if start_batch_idx is None or batch_idx >= start_batch_idx:
                    # Convert audio to float
                    if not isinstance(batch['audio'][0][0], float):
                        if len(batch['audio'][0]) == student_asr:
                            batch['audio'] = pcm2float(batch['audio'], dtype='float32')
                        else:
                            batch['audio'] = resample(pcm2float(batch['audio'], dtype='float32'), sr_orig=48000,
                                                    sr_new=student_asr) 
                    else:
                        batch['audio'] = np.array(batch['audio'])[:, np.newaxis, :]

                    if melSpec:
                        X = [get_melspectrogram(
                                            batch['audio'][i].flatten(), 
                                            n_fft=n_fft, mel_hop_length=n_hop,
                                            samp_rate=student_asr, n_mels=n_mels, 
                                            fmax=fmax
                                        ) for i in range(batch_size)
                            ]

                        batch['audio'] = np.array(X)[:, :, :, np.newaxis]
                    yield batch

                batch_idx += 1
                curr_batch_size = 0
                batch = None


def single_epoch_data_generator(data_dir, emb_dir, emb_key, epoch_size, **kwargs):
    while True:
        data_gen = data_generator(data_dir, emb_dir, emb_key, **kwargs)

        for idx, item in enumerate(data_gen):
            yield item
            # Once we generate all batches for an epoch, restart the generator
            if (idx + 1) == epoch_size:
                break

def get_dir_splits(data_dir, split=0.017):
    all_files = os.listdir(os.path.abspath(data_dir))
    #Make sure only h5 files are retrieved
    data_files = list(filter(lambda file: file.endswith('.h5'), all_files))
    
    random.shuffle(data_files)
    split_index = int(len(data_files) * split)
    train_files = data_files[:split_index]
    val_files = data_files[split_index:]
    
    return train_files, val_files

def train(train_data_dir, validation_data_dir, emb_train_dir, emb_valid_dir, output_dir,
          student_weight_path=None, approx_mode='umap', 
          approx_train_size=None, neighbors=None, min_dist=None, metric='euclidean', pca_kernel='linear',
          num_epochs=300, learning_rate=0.00001, 
          train_epoch_size=4096, validation_epoch_size=1024, 
          train_batch_size=64, validation_batch_size=64,
          model_type='cnn_L3_melspec2', n_mels=64, n_hop=160, n_dft=1024, samp_rate=8000, 
          fmax=None, halved_convs=True, melSpec=False,
          log_path=None, disable_logging=False, random_state=20180216, 
          verbose=True, checkpoint_interval=10, gpus=1, 
          continue_model_dir=None, gsheet_id=None, google_dev_app_name=None):

    init_console_logger(LOGGER, verbose=verbose)
    if not disable_logging:
        init_file_logger(LOGGER, log_path=log_path)

    LOGGER.debug('Initialized logging.')
    LOGGER.info('Embedding Reduction Mode: %s', approx_mode)
    
    model_desc = ''
    if continue_model_dir:
        model_desc = continue_model_dir.split('/')[-2]
        latest_model_path = os.path.join(continue_model_dir, 'model_latest.h5')
        student_base_model = keras.models.load_model(latest_model_path, custom_objects={'Melspectrogram': Melspectrogram})
        student_samp_rate, n_mels, n_hop, n_dft, fmax = get_model_params(model_desc, continue_train=True)
        
    elif student_weight_path:
        model_desc = os.path.basename(student_weight_path).strip('.h5')
        student_base_model = get_student_model(student_weight_path)
        student_samp_rate, n_mels, n_hop, n_dft, fmax = get_model_params(model_desc)
        
    else:
        student_samp_rate = samp_rate
        student_base_model, inputs, outputs = construct_cnn_L3_melspec2_audio_model(
                                                    n_mels=n_mels, 
                                                    n_hop=n_hop, 
                                                    n_dft=n_dft,
                                                    asr=samp_rate, 
                                                    fmax=fmax, 
                                                    halved_convs=halved_convs
                                                )
              
    if halved_convs or 'half' in model_desc:
        model_repr = str(student_samp_rate)+'_'+str(n_mels)+'_'+str(n_hop)+'_'+str(n_dft)+'_half'+'_fmax_'+str(fmax)
    else:
        model_repr = str(student_samp_rate)+'_'+str(n_mels)+'_'+str(n_hop)+'_'+str(n_dft)+'_fmax_'+str(fmax)

    emb_key = None
    if continue_model_dir:
        model_dir = continue_model_dir
        emb_key = continue_model_dir.split('/')[-3]
    else:
        if 'music' in train_data_dir:
            dataset = 'music'
        elif 'environmental' in train_data_dir:
            dataset = 'env'
        else:
            dataset = 'sonyc'
        
        if approx_mode == 'umap' or approx_mode == 'pca': 
            # Ex. of train_data_dir: $SCRATCH/reduced_embeddings/sonyc/pca/dpp/day/500000/pca_ndata=500000_emb=256_kernel=linear
            mode_idx = emb_train_dir.find(approx_mode)
            model_attribute = emb_train_dir[mode_idx:]
            emb_key = model_attribute.split('/')[-1]    #Ex. pca_ndata=500000_emb=256_kernel=linear

        elif approx_mode == 'mse':
            model_attribute = 'mse_original'
            emb_key = 'l3_embedding'

        else:
            raise ValueError('Invalid approximation mode: {}'.format(approx_mode))

        model_dir = os.path.join(
            output_dir,  
            dataset,  
            model_attribute,
            model_repr,
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )
        
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    student_emb_len = get_embedding_length(student_base_model);  
    LOGGER.info('Student sampling rate: {}'.format(student_samp_rate))
    LOGGER.info('Model Representation: {}'.format(model_repr))
    LOGGER.info('Model Attribute: {}'.format(model_attribute))
    LOGGER.info('Student Embedding Length: {}'.format(student_emb_len))

    # emb_key = get_embedding_key(
    #     approx_mode, 
    #     approx_train_size, 
    #     student_emb_length, 
    #     neighbors=neighbors, 
    #     metric=metric, 
    #     min_dist=min_dist, 
    #     pca_kernel=pca_kernel)

    param_dict = {
        'username': getpass.getuser(),
        'model_dir': model_dir,
        'train_data_dir': train_data_dir,
        'validation_data_dir': validation_data_dir,
        'reduced_emb_train_dir': emb_train_dir,
        'reduced_emb_valid_dir': emb_valid_dir,
        'approx_mode': approx_mode,
        'emb_key': emb_key,
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
        'git_commit': git.Repo(os.path.dirname(os.path.abspath(__file__)),
                               search_parent_directories=True).head.object.hexsha,
        'gsheet_id': gsheet_id,
        'google_dev_app_name': google_dev_app_name
    }
         
    LOGGER.info('Training with the following arguments: {}'.format(param_dict))

    #Convert the base (single-GPU) model to Multi-GPU model
    model = multi_gpu_model(student_base_model, gpus=gpus)

    train_config_path = os.path.join(model_dir, 'config.json')
    with open(train_config_path, 'w') as fd:
        json.dump(param_dict, fd, indent=2)

    LOGGER.info('Compiling model...')
    model.compile(Adam(lr=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mae'])
    
    LOGGER.info('Model files can be found in "{}"'.format(model_dir))

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

    # Set up callbacks
    cb = []
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

    #earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10)
    reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

    #cb.append(earlyStopping)
    cb.append(reduceLR)

    if gsheet_id:
        cb.append(GSheetLogger(google_dev_app_name, gsheet_id, param_dict))

    LOGGER.info('Setting up train data generator...')
    if continue_model_dir is not None:
        train_start_batch_idx = train_epoch_size * (last_epoch_idx + 1)
    else:
        train_start_batch_idx = None

    if train_data_dir == validation_data_dir:
        LOGGER.info('The dataset is not split into train and valid. Splitting the train data!')
        emb_train_dir, emb_valid_dir = get_dir_splits(emb_train_dir)
    
    train_gen = data_generator(train_data_dir,
                               emb_train_dir,
                               emb_key,
                               student_asr=student_samp_rate,
                               batch_size=train_batch_size,
                               n_fft=n_dft, n_mels=n_mels, n_hop=n_hop, fmax=fmax,
                               random_state=random_state,
                               melSpec=melSpec,
                               start_batch_idx=train_start_batch_idx)

    val_gen = single_epoch_data_generator(validation_data_dir,
                                          emb_valid_dir,
                                          emb_key,
                                          validation_epoch_size,
                                          student_asr=student_samp_rate,
                                          batch_size=validation_batch_size,
                                          n_fft=n_dft, n_mels=n_mels, n_hop=n_hop, fmax=fmax,
                                          melSpec=melSpec,
                                          random_state=random_state)

    train_gen = pescador.maps.keras_tuples(train_gen,
                                           'audio',
                                           'label')

    val_gen = pescador.maps.keras_tuples(val_gen,
                                         'audio',
                                         'label')

    # Fit the model
    LOGGER.info('Fitting model...')
    verbosity = 1 if verbose else 2

    if continue_model_dir is not None:
        initial_epoch = last_epoch_idx + 1
    else:
        initial_epoch = 0

    history = model.fit_generator(train_gen, train_epoch_size, num_epochs,
                                  validation_data=val_gen,
                                  validation_steps=validation_epoch_size,
                                  callbacks=cb,
                                  verbose=verbosity,
                                  initial_epoch=initial_epoch)
    
    LOGGER.info('Done training. Saving results to disk...')
    # Save history
    with open(os.path.join(model_dir, 'history.pkl'), 'wb') as fd:
        pickle.dump(history.history, fd)

    LOGGER.info('Done!')
