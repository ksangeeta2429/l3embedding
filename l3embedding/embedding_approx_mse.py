import getpass
import git
import os
import random
import csv
import datetime
import json
import pickle
import csv
import numpy as np
import keras
import pescador
import tensorflow as tf
import h5py
import copy
import tempfile
import umap
from keras import backend as K
from keras import activations
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from skimage import img_as_float
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from .training_utils import conv_dict_to_val_list, multi_gpu_model
from .model import *
from .audio import pcm2float
from gsheets import get_credentials, append_row, update_experiment, get_row
from googleapiclient import discovery
from log import *
from kapre.time_frequency import Spectrogram, Melspectrogram
from resampy import resample
from sklearn.manifold import TSNE


LOGGER = logging.getLogger('embedding_approx_mse')
LOGGER.setLevel(logging.DEBUG)


graph = tf.get_default_graph()
weight_path = 'models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5'
audio_model = load_embedding(weight_path, model_type = 'cnn_L3_melspec2', embedding_type = 'audio', \
                             pooling_type = 'short', kd_model=False, tgt_num_gpus = 1)

def get_embedding_length(model):
    embed_layer = model.get_layer('audio_embedding_layer')
    pool_size = tuple(embed_layer.get_output_shape_at(0)[1:3])
    y_a = MaxPooling2D(pool_size=pool_size, padding='same')(embed_layer.output)
    y_a = Flatten()(y_a)
    emb_len = y_a.get_shape().as_list()[1]
    return emb_len


def data_generator(data_dir, approx_mode='umap', neighbors=10, min_dist=0.5, student_emb_length=None, \
                   batch_size=512, random_state=20180216, start_batch_idx=None, keys=None):
    
    if student_emb_length is None:
        raise ValueError('Student Embedding Length should be given for Embedding Approximation')

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
                    # Convert audio to float
                    batch['audio'] = pcm2float(batch['audio'], dtype='float32')
                        
                    # Get the embedding layer output from the audio_model and flatten it to be treated as labels for the student audio model
                    with graph.as_default():
                        embeddings = audio_model.predict(batch['audio'])
                        if approx_mode == 'umap':
                            batch['label'] = umap.umap_.UMAP(n_neighbors=neighbors, min_dist=min_dist, \
                                                             n_components=student_emb_length).fit_transform(embeddings)
                        else if approx_mode == 'tsne':
                            batch['label'] = TSNE(perplexity=neighbors, n_components=student_emb_length, n_iter=300).fit_transform(embeddings)
                                                
                    yield batch

                batch_idx += 1
                curr_batch_size = 0
                batch = None


def single_epoch_data_generator(data_dir, approx_mode='umap', neighbors=10, min_dist=0.5, student_emb_length,\
                                batch_size=512, **kwargs):
    while True:
        data_gen = data_generator(data_dir, approx_mode=approx_mode, neighbors=neighbors, min_dist=min_dist, \
                                  student_emb_length=student_emb_length, batch_size=batch_size, **kwargs)
        for idx, item in enumerate(data_gen):
            yield item
            # Once we generate all batches for an epoch, restart the generator
            if (idx + 1) == epoch_size:
                break


def train(train_data_dir, validation_data_dir, output_dir, student_weight_path=None, approx_mode='umap', neighbors=10, min_dist=None,\
          num_epochs=300, train_epoch_size=4096, validation_epoch_size=1024, train_batch_size=64, validation_batch_size=64,\
          model_type='cnn_L3_melspec2', n_mels=256, n_hop=242, n_dft=2048, samp_rate=48000, fmax=None, halved_convs=False,\
          log_path=None, disable_logging=False, random_state=20180216,learning_rate=0.00001, verbose=True, checkpoint_interval=10,\
          gpus=1, continue_model_dir=None, gsheet_id=None, google_dev_app_name=None, thresholds=None):


    init_console_logger(LOGGER, verbose=verbose)
    if not disable_logging:
        init_file_logger(LOGGER, log_path=log_path)
    LOGGER.debug('Initialized logging.')
    LOGGER.info('Embedding Reduction Mode: %s', approx_mode)
    LOGGER.info('Neighbors/Perplexity: %s', neighbors)
    
    if approx_mode == 'umap':
        if min_dist is None:
            min_dist = 0.5
        model_attribute = 'emb_approx_umap_k_' + str(neighbors) + '_dist_' + str(min_dist)
    if approx_mode == 'tsne':
        model_attribute = 'emb_approx_tsne_k_' + str(neighbors)
    
    
    LOGGER.info('Student sampling rate: {}'.format(student_samp_rate))

    # Make sure the directories we need exist
    if continue_model_dir:
        model_dir = continue_model_dir
    else:
        model_dir = os.path.join(output_dir, 'embedding_approx', model_attribute, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        
    if continue_model_dir:
        latest_model_path = os.path.join(continue_model_dir, 'model_latest.h5')
        student_base_model = keras.models.load_model(latest_model_path, custom_objects={'Melspectrogram': Melspectrogram})
    else if student_weight_path:
        model_repr = weight_path
        student_base_model = keras.models.load_model(student_weight_path, custom_objects={'Melspectrogram': Melspectrogram})
    else:
        student_base_model, inputs, outputs = MODELS[model_type](n_mels=n_mels, n_hop=n_hop, n_dft=n_dft,
                                                                 asr=samp_rate, fmax=fmax, halved_convs=halved_convs, num_gpus=1)
        if halved_convs:
            model_repr = str(samp_rate)+'_'+str(n_mels)+'_'+str(n_hop)+'_'+str(n_dft)+'_half'+'_fmax_'+str(fmax)
        else:
            model_repr = str(samp_rate)+'_'+str(n_mels)+'_'+str(n_hop)+'_'+str(n_dft)+'_fmax_'+str(fmax)

    student_emb_len = get_embedding_length(student_base_model);

    param_dict = {
        'username': getpass.getuser(),
        'train_data_dir': train_data_dir,
        'validation_data_dir': validation_data_dir,
        'output_dir': output_dir,
        'approx_mode': approx_mode,
        'neighbors': neighbors,
        'min_dist': min_dist,
        'model_repr': model_repr,
        'student_emb_len': student_emb_len, 
        'num_epochs': num_epochs,
        'train_epoch_size': train_epoch_size,
        'validation_epoch_size': validation_epoch_size,
        'train_batch_size': train_batch_size,
        'validation_batch_size': validation_batch_size,
        'model_type': model_type,
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

    param_dict['model_dir'] = model_dir
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

    best_val_loss_cb = MultiGPUCheckpointCallback(best_valid_loss_weight_path,
                                                  student_base_model,
                                                  save_weights_only=True,
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

    if gsheet_id:
        cb.append(GSheetLogger(google_dev_app_name, gsheet_id, param_dict, loss_type))

    LOGGER.info('Setting up train data generator...')
    if continue_model_dir is not None:
        train_start_batch_idx = train_epoch_size * (last_epoch_idx + 1)
    else:
        train_start_batch_idx = None


    train_gen = data_generator(train_data_dir,
                               approx_mode=approx_mode,
                               neighbors=neighbors,
                               min_dist=min_dist,
                               student_emb_length=student_emb_len,
                               batch_size=train_batch_size,
                               random_state=random_state,
                               start_batch_idx=train_start_batch_idx)

    val_gen = single_epoch_data_generator(validation_data_dir,
                                          approx_mode=approx_mode,
                                          neighbors=neighbors,
                                          min_dist=min_dist,
                                          student_emb_length=student_emb_len,
                                          batch_size=validation_batch_size,
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
