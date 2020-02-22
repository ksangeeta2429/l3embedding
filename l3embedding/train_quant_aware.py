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
import copy
import numpy as np
import keras
import pescador
import tensorflow as tf
import h5py
import librosa
from tqdm.keras import TqdmCallback
from keras import backend as K
from skimage import img_as_float
from keras import activations
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model
from gsheets import get_credentials, append_row, update_experiment, get_row
from .training_utils import conv_dict_to_val_list, multi_gpu_model, MultiGPUCheckpointCallback
from .model import *
from .audio import pcm2float
from log import *
from kapre.time_frequency import Spectrogram, Melspectrogram
from resampy import resample
import sys
from googleapiclient import discovery

LOGGER = logging.getLogger('l3embedding')
LOGGER.setLevel(logging.DEBUG)

class LossHistory(keras.callbacks.Callback):
    """
    Keras callback to record loss history
    """

    def __init__(self, outfile):
        super().__init__()
        self.outfile = outfile

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.loss = []
        self.val_loss = []

    # def on_batch_end(self, batch, logs={}):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        loss_dict = {'loss': self.loss, 'val_loss': self.val_loss}
        with open(self.outfile, 'wb') as fp:
            pickle.dump(loss_dict, fp)

class GSheetLogger(keras.callbacks.Callback):
    """
    Keras callback to update Google Sheets Spreadsheet
    """

    def __init__(self, google_dev_app_name, spreadsheet_id, param_dict):
        super(GSheetLogger).__init__()
        self.google_dev_app_name = google_dev_app_name
        self.spreadsheet_id = spreadsheet_id
        self.credentials = get_credentials(google_dev_app_name)
        self.service = discovery.build('sheets', 'v4', credentials=self.credentials, cache_discovery=False)
        self.param_dict = copy.deepcopy(param_dict)

        row_num = get_row(self.service, self.spreadsheet_id, self.param_dict, 'embedding')
        if row_num is None:
            append_row(self.service, self.spreadsheet_id, self.param_dict, 'embedding')

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.best_train_loss = float('inf')
        self.best_valid_loss = float('inf')
        self.best_train_acc = float('-inf')
        self.best_valid_acc = float('-inf')

    # def on_batch_end(self, batch, logs={}):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        latest_epoch = epoch
        latest_train_loss = logs.get('loss')
        latest_valid_loss = logs.get('val_loss')
        latest_train_acc = logs.get('acc')
        latest_valid_acc = logs.get('val_acc')

        if latest_train_loss < self.best_train_loss:
            self.best_train_loss = latest_train_loss
        if latest_valid_loss < self.best_valid_loss:
            self.best_valid_loss = latest_valid_loss
        if latest_train_acc > self.best_train_acc:
            self.best_train_acc = latest_train_acc
        if latest_valid_acc > self.best_valid_acc:
            self.best_valid_acc = latest_valid_acc

        values = [
            latest_epoch, latest_train_loss, latest_valid_loss,
            latest_train_acc, latest_valid_acc, self.best_train_loss,
            self.best_valid_loss, self.best_train_acc, self.best_valid_acc]

        update_experiment(self.service, self.spreadsheet_id, self.param_dict, 'V', 'AD', values, 'embedding')
                          #'R', 'Z', values, 'embedding')


class TimeHistory(keras.callbacks.Callback):
    """
    Keras callback to log epoch and batch running time
    """
    # Copied from https://stackoverflow.com/a/43186440/1260544
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.batch_times = []

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs=None):
        t = time.time() - self.epoch_time_start
        LOGGER.info('Epoch took {} seconds'.format(t))
        self.epoch_times.append(t)

    def on_batch_begin(self, batch, logs=None):
        self.batch_time_start = time.time()

    def on_batch_end(self, batch, logs=None):
        t = time.time() - self.batch_time_start
        #LOGGER.info('Batch took {} seconds'.format(t))
        self.batch_times.append(t)

def get_restart_info(history_path):
    last = None
    with open(history_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row

    return int(last['epoch']), float(last['val_acc']), float(last['val_loss'])

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
            
def amplitude_to_db(S, amin=1e-10, dynamic_range=80.0):
    magnitude = np.abs(S)
    power = np.square(magnitude, out=magnitude)
    ref_value = power.max()

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= log_spec.max()

    log_spec = np.maximum(log_spec, -dynamic_range)
    return log_spec

def get_melspectrogram(frame, n_fft=2048, mel_hop_length=242, samp_rate=48000, n_mels=256,\
                       quant_melspec=False, fmax=None):
    S = np.abs(librosa.core.stft(frame, n_fft=n_fft, hop_length=mel_hop_length, window='hann', center=True, pad_mode='constant'))
    S = librosa.feature.melspectrogram(sr=samp_rate, S=S, n_fft=n_fft, n_mels=n_mels, fmax=fmax, power=1.0, htk=True)
    S = amplitude_to_db(np.array(S))
    return S

def cycle_shuffle(iterable, shuffle=True):
    lst = list(iterable)
    while True:
        yield from lst
        if shuffle:
            random.shuffle(lst)

def data_generator(data_dir, batch_size=512, random_state=20180123, samp_rate=48000,
                   n_mels=64, n_hop=160, n_dft=1064,  start_batch_idx=None, keys=None):
    
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
                X  = []
                if start_batch_idx is None or batch_idx >= start_batch_idx:
                    # Preprocess video so samples are in [-1,1]
                    batch['video'] = 2 * img_as_float(batch['video']).astype('float32') - 1

                    # Convert audio to float
                    if(samp_rate==48000):
                        batch['audio'] = pcm2float(batch['audio'], dtype='float32')
                    else:
                        batch['audio'] = resample(pcm2float(batch['audio'], dtype='float32'), sr_orig=48000,
                                                  sr_new=samp_rate)
                        
                    X = [get_melspectrogram(x.flatten(), n_fft=n_dft, mel_hop_length=n_hop,
                                            samp_rate=samp_rate, n_mels=n_mels) for x in batch['audio']]
                    
                    batch['audio'] = np.array(X)[:, :, :, np.newaxis]
                    #print('Shape of audio batch:', np.shape(batch['audio'])) (64, 64, 51, 1)
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
                
def restore_save_quantized_model(model_path, output_dir):
    
    K.clear_session()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    output_path = os.path.join(output_dir, 'frozen_model_quant.pb')
    eval_graph = tf.Graph()
    eval_sess = tf.Session(config=config, graph=eval_graph)
    
    K.set_session(eval_sess)
    
    with eval_graph.as_default():
        K.set_learning_phase(0)
        eval_model = keras.models.load_model(model_path)
        #print(eval_model.summary())
        
        tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
        eval_sess.run(tf.global_variables_initializer())
        
        eval_graph_def = eval_graph.as_graph_def()
        saver = tf.train.Saver()
        saver.restore(eval_sess, os.path.join(output_dir, os.path.basename(output_dir)))

        print(eval_model.input.op.name)
        print(eval_model.output.op.name)
        
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                                                                        eval_sess,
                                                                        eval_graph_def,
                                                                        [eval_model.output.op.name]
                                                                        )

        with open(output_path, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())
            
def train(train_data_dir, validation_data_dir, output_dir, num_epochs=1, 
          train_epoch_size=512, validation_epoch_size=1024, train_batch_size=64,
          validation_batch_size=64, model_type='cnn_L3_nomelspec', random_state=20180123,
          learning_rate=1e-4, verbose=False, checkpoint_interval=10, n_mels=64, n_hop=160, n_dft=1024,
          samp_rate=8000, fmax=None, halved_convs=True, log_path=None, disable_logging=False, 
          gpus=2, continue_model_dir=None, gsheet_id=None, google_dev_app_name=None):

    init_console_logger(LOGGER, verbose=verbose)
    if not disable_logging:
        init_file_logger(LOGGER, log_path=log_path)
    LOGGER.debug('Initialized logging.')

    # Form model ID
    data_subset_name = os.path.basename(train_data_dir)
    data_subset_name = data_subset_name[:data_subset_name.rindex('_')]
    model_id = os.path.join(data_subset_name, model_type + '_quant')

    param_dict = {
          'username': getpass.getuser(),
          'train_data_dir': train_data_dir,
          'validation_data_dir': validation_data_dir,
          'model_id': model_id,
          'output_dir': output_dir,
          'samp_rate': int(samp_rate/1000),
          'num_mels': n_mels,
          'num_hops': int(np.ceil(samp_rate/n_hop)),
          'num_dft': n_dft,
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
          'git_commit': git.Repo(os.path.dirname(os.path.abspath(__file__)), search_parent_directories=True).head.object.hexsha,
          'gsheet_id': gsheet_id,
          'google_dev_app_name': google_dev_app_name
    }
    LOGGER.info('Training with the following arguments: {}'.format(param_dict))

    # Make sure the directories we need exist
    if continue_model_dir:
        model_dir = continue_model_dir
    else:
        model_dir = os.path.join(output_dir, 'embedding', model_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    LOGGER.info('Model files can be found in "{}"'.format(model_dir))

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

    latest_weight_path = os.path.join(model_dir, 'model_latest.h5')
    best_valid_acc_weight_path = os.path.join(model_dir, 'model_best_valid_accuracy.h5')
    best_valid_loss_weight_path = os.path.join(model_dir, 'model_best_valid_loss.h5')
    checkpoint_weight_path = os.path.join(model_dir, 'model_checkpoint.{epoch:02d}.h5')

    # Load information about last epoch for initializing callbacks and data generators
    if continue_model_dir is not None:
        prev_train_hist_path = os.path.join(continue_model_dir, 'history_csvlog.csv')
        last_epoch_idx, last_val_acc, last_val_loss = get_restart_info(prev_train_hist_path)

    LOGGER.info('Setting up train data generator...')
    if continue_model_dir is not None:
        train_start_batch_idx = train_epoch_size * (last_epoch_idx + 1)
    else:
        train_start_batch_idx = None

    train_gen = data_generator(train_data_dir,
                               batch_size=train_batch_size,
                               random_state=random_state,
                               start_batch_idx=train_start_batch_idx,
                               samp_rate=samp_rate,
                               n_mels=n_mels,
                               n_hop=n_hop,
                               n_dft=n_dft)

    train_gen = pescador.maps.keras_tuples(train_gen,
                                           ['video', 'audio'],
                                           'label')

    LOGGER.info('Setting up validation data generator...')
    val_gen = single_epoch_data_generator(validation_data_dir,
                                          validation_epoch_size,
                                          batch_size=validation_batch_size,
                                          random_state=random_state,
                                          samp_rate=samp_rate,
                                          n_mels=n_mels,
                                          n_hop=n_hop,
                                          n_dft=n_dft)

    val_gen = pescador.maps.keras_tuples(val_gen,
                                         ['video', 'audio'],
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

    K.clear_session()
    
    #train graph
    train_graph = tf.Graph()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    train_sess = tf.Session(config=config, graph=train_graph)
    K.set_session(train_sess)
    
    with train_graph.as_default():       
        if continue_model_dir:
            latest_model_path = os.path.join(continue_model_dir, 'model_latest.h5')
            m = keras.models.load_model(latest_model_path)
        else:
            m, inputs, outputs = MODELS[model_type](n_mels=n_mels, n_hop=n_hop, n_dft=n_dft,
                                                    asr=samp_rate, fmax=fmax, halved_convs=halved_convs, num_gpus=1)
       
        # Save the model
        model_spec_path = os.path.join(model_dir, 'model_spec.pkl')
        model_spec = keras.utils.serialize_keras_object(m)
        with open(model_spec_path, 'wb') as fd:
            pickle.dump(model_spec, fd)
            
        model_json_path = os.path.join(model_dir, 'model.json')
        model_json = m.to_json()
        with open(model_json_path, 'w') as fd:
            json.dump(model_json, fd, indent=2)
            
        # Set up callbacks
        cb = []

        cb.append(MultiGPUCheckpointCallback(latest_weight_path,
                                             m,
                                             save_weights_only=False,
                                             verbose=1))

        best_val_acc_cb = MultiGPUCheckpointCallback(best_valid_acc_weight_path,
                                                     m,
                                                     save_weights_only=False,\
                                                     save_best_only=True,\
                                                     verbose=1,
                                                     monitor='val_acc')
            
        if continue_model_dir is not None:
            best_val_acc_cb.best = last_val_acc
        cb.append(best_val_acc_cb)

        # Callback for multi-gpu model
        best_val_loss_cb = MultiGPUCheckpointCallback(best_valid_loss_weight_path,
                                                      m,
                                                      save_weights_only=False,
                                                      save_best_only=True,
                                                      verbose=1,
                                                      monitor='val_loss')
            
        if continue_model_dir is not None:
            best_val_loss_cb.best = last_val_loss
        cb.append(best_val_loss_cb)

        checkpoint_cb = keras.callbacks.ModelCheckpoint(checkpoint_weight_path,
                                                        save_weights_only=True,
                                                        period=checkpoint_interval)
        
        if continue_model_dir is not None:
            checkpoint_cb.epochs_since_last_save = (last_epoch_idx + 1) % checkpoint_interval
        cb.append(checkpoint_cb)

        if gsheet_id:
            cb.append(GSheetLogger(google_dev_app_name, gsheet_id, param_dict))
        
        timer_cb = TimeHistory()
        cb.append(timer_cb)

        history_checkpoint = os.path.join(model_dir, 'history_checkpoint.pkl')
        cb.append(LossHistory(history_checkpoint))

        history_csvlog = os.path.join(model_dir, 'history_csvlog.csv')
        cb.append(keras.callbacks.CSVLogger(history_csvlog, append=True,
                                            separator=','))

        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10)
        reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

        cb.append(earlyStopping)
        cb.append(reduceLR)
        cb.append(TqdmCallback(epochs=num_epochs, batch_size=train_batch_size, verbose=1))
        
        #Convert the base (single-GPU) model to Multi-GPU model
        if gpus == 1:
            model = m
        else:
            model = multi_gpu_model(m, gpus=gpus)
            
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    
        LOGGER.info('Compiling model...')
        model.compile(Adam(lr=learning_rate), loss=loss, metrics=metrics)
        
        tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=50)
        initialize_uninitialized_variables(train_sess)
        
        history = model.fit_generator(train_gen, train_epoch_size, num_epochs,
                                      validation_data=val_gen,
                                      validation_steps=validation_epoch_size,
                                      callbacks=cb,
                                      verbose=0, #verbosity,
                                      initial_epoch=initial_epoch)

        #save graph and checkpoints
        saver = tf.train.Saver()
        saver.save(train_sess, save_path=os.path.join(model_dir, os.path.basename(model_dir)))

        LOGGER.info('Done training. Saving results to disk...')
        # Save history
        with open(os.path.join(model_dir, 'history.pkl'), 'wb') as fd:
            pickle.dump(history.history, fd)

        #restore_save_quantized_model(best_valid_loss_weight_path, model_dir)
        
    LOGGER.info('Done!')
