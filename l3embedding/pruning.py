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
from keras.regularizers import l2
import tensorflow as tf
import keras
from keras.optimizers import Adam
import pescador
from keras.layers import *
from .audio import pcm2float
import h5py
from keras.models import Model
from .model import *
from keras.optimizers import Adam
import pescador
from skimage import img_as_float
from keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import copy
from gsheets import get_credentials, append_row, update_experiment, get_row
from googleapiclient import discovery
from log import *

LOGGER = logging.getLogger('prunedl3embedding')
LOGGER.setLevel(logging.DEBUG)


##########
# Pruning Version 1:
# Step 1: Prune each layer with different aprsity levels to decide the sensitivity of each 
#         Prune whole model after deciding the sensitivity of each layer and test the performance
# Step 2: Fine-tune the whole model at a go
# Step 3: Use knowledge distillation setting to retrain the pruned model. This does not take into account the AVC problem.

##########
# Step 1
# Pseudo-code
# Construct the l3 model and load the weights from trained cnn_l3_melspec2
# Version 1
# Run a loop for different sparsity values [30, 40, 50, 60, 70] for each CONV block and test the performance
# Version 2
# Run a loop for different sparsity values [30, 40, 50, 60, 70, 80, 85, 90, 95] for each CONV layer and test the performance
##########

##########
# Pruning Version 2: Iterative pruning
# Prune and fine-tune each layer at a time
##########

##########
# Pruning Version 3: Prune whole feature-maps
# Pseudo-code
# Step 1: Get the magnitude of each filter
# Step 2: Experiment with different sparsity values and drop the resulting insignificant filters (with magnitude lesser than the threshold)
# Step 3: Drop the channels corresponding to the dropped filters in the layer's output
# Step 4: Form the new architecture according to the new number of filters
# Step 5: Freeze the video model and fine-tune the audio and the merged Dense layers
##########

graph = tf.get_default_graph()
weight_path = 'models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5'
audio_model = load_embedding(weight_path, model_type = 'cnn_L3_melspec2', embedding_type = 'audio', \
                             pooling_type = 'short', kd_model=False, tgt_num_gpus = 1)

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


def cycle_shuffle(iterable, shuffle=True):
    lst = list(iterable)
    while True:
        yield from lst
        if shuffle:
            random.shuffle(lst)

class GSheetLogger(keras.callbacks.Callback):
    """
    Keras callback to update Google Sheets Spreadsheet
    """

    def __init__(self, google_dev_app_name, spreadsheet_id, param_dict):
        super(GSheetLogger).__init__()
        self.google_dev_app_name = google_dev_app_name
        self.spreadsheet_id = spreadsheet_id
        self.credentials = get_credentials(google_dev_app_name)
        self.service = discovery.build('sheets', 'v4', credentials=self.credentials)
        self.param_dict = copy.deepcopy(param_dict)

        row_num = get_row(self.service, self.spreadsheet_id, self.param_dict, 'prunedembedding')
        if row_num is None:
            append_row(self.service, self.spreadsheet_id, self.param_dict, 'prunedembedding')

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

        update_experiment(self.service, self.spreadsheet_id, self.param_dict,
                          'R', 'Z', values, 'prunedembedding')


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
        LOGGER.info('Batch took {} seconds'.format(t))
        self.batch_times.append(t)


def data_generator(data_dir, kd_model=False, batch_size=512, random_state=20180216, start_batch_idx=None, keys=None):
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
        if kd_model:
            keys = ['audio']
        else:
            keys = ['audio', 'video', 'label']


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
                    if not kd_model:
                        # Preprocess video so samples are in [-1,1]
                        batch['video'] = 2 * img_as_float(batch['video']).astype('float32') - 1

                    # Convert audio to float
                    batch['audio'] = pcm2float(batch['audio'], dtype='float32')
                        
                    if kd_model:
                        # Get the embedding layer output from the audio_model and flatten it to be treated as labels for the student audio model
                        with graph.as_default():
                            batch['label'] = audio_model.predict(batch['audio'])
                                                
                    yield batch

                batch_idx += 1
                curr_batch_size = 0
                batch = None


def single_epoch_data_generator(data_dir, epoch_size, kd_model, **kwargs):
    while True:
        data_gen = data_generator(data_dir, kd_model, **kwargs)
        for idx, item in enumerate(data_gen):
            yield item
            # Once we generate all batches for an epoch, restart the generator
            if (idx + 1) == epoch_size:
                break


def get_sparsity_blocks(block, sparsity_blk, sparsity_value_blocks):
    num_conv_blocks = 4
    sparsity = {}

    for blockid in range(num_conv_blocks):
        block_layer_1 = 'conv_'+str(2*blockid + 1)
        block_layer_2 = 'conv_'+str(2*blockid + 2)
        if (blockid == block):
            sparsity[block_layer_1] = sparsity_blk
            sparsity[block_layer_2] = sparsity_blk
        else:
            sparsity[block_layer_1] = sparsity_value_blocks[blockid]
            sparsity[block_layer_2] = sparsity_value_blocks[blockid]
    
    return sparsity


def get_sparsity_layers(layer, sparsity_layer, sparsity_value_layers):
    sparsity = {}
    for idx in range(len(sparsity_value_layers)):
        layer_name = 'conv_'+str(idx+1)
        if(layer and sparsity_layer and idx == layer):
            sparsity[layer_name] = sparsity_layer
        else:
            sparsity[layer_name] = sparsity_value_layers[idx]

    return sparsity


def calculate_threshold(weights, ratio):
    return tf.contrib.distributions.percentile(K.abs(weights), ratio)

        
def sparsify_layer(model, sparsity_dict):
    masks = {}
    for layer in model.layers:
        if 'conv_' in layer.name:
            target_weights = np.empty_like(layer.get_weights())
            weights = layer.get_weights()[0]

            if sparsity_dict[layer.name]:
                threshold = calculate_threshold(weights, sparsity_dict[layer.name])
                mask      = K.cast(K.greater(K.abs(weights), threshold), dtypes.float32)
                new_weights = weights * K.eval(mask)
                target_weights[0] = new_weights
            
                target_weights[1] = layer.get_weights()[1]
                layer.set_weights(target_weights)

            else:
                mask = K.ones_like(weights)

            masks[layer.name] = mask
            
    return model, masks


def sparsify_block(model, sparsity_dict):
    conv_blocks = 4

    for block in range(conv_blocks):
        layer_1 = 'conv_'+str(2*block + 1)
        layer_2 = 'conv_'+str(2*block + 2)
        
        if (sparsity_dict[layer_1]):
            weights_1 = model.get_layer(layer_1).get_weights()[0]
            weights_2 = model.get_layer(layer_2).get_weights()[0]
            weights = np.append(weights_1.flatten(), weights_2.flatten())
            
            threshold = calculate_threshold(weights, sparsity_dict[layer_1])

            #print(K.eval(threshold))
            target_weights_1 = np.empty_like(model.get_layer(layer_1).get_weights())
            target_weights_2 = np.empty_like(model.get_layer(layer_2).get_weights())

            mask_1      = K.cast(K.greater(K.abs(weights_1), threshold), dtypes.float32)
            mask_2      = K.cast(K.greater(K.abs(weights_2), threshold), dtypes.float32)

            new_weights_1 = weights_1 * K.eval(mask_1)
            new_weights_2 = weights_2 * K.eval(mask_2)
            
            target_weights_1[0] = new_weights_1
            target_weights_1[1] = model.get_layer(layer_1).get_weights()[1]

            target_weights_2[0] = new_weights_2
            target_weights_2[1] = model.get_layer(layer_2).get_weights()[1]
            
            model.get_layer(layer_1).set_weights(target_weights_1)
            model.get_layer(layer_2).set_weights(target_weights_2)
            
            #print(new_weights)
                        
    return model


def load_student_audio_model_withFFT(include_layers, num_filters = [64, 64, 128, 128, 256, 256, 512, 512]):
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
    filt_size_a_1 = (3, 3)
    pool_size_a_1 = (2, 2)
    
    if include_layers[0]:
        y_a = Conv2D(num_filters[0], filt_size_a_1, padding='same',
                     kernel_initializer='he_normal',
                     name='conv_1',
                     kernel_regularizer=regularizers.l2(weight_decay))(y_a)
        y_a = BatchNormalization()(y_a)
        y_a = Activation('relu')(y_a)

    if include_layers[1]:
        y_a = Conv2D(num_filters[1], filt_size_a_1, padding='same',
                     kernel_initializer='he_normal',
                     name='conv_2',
                     kernel_regularizer=regularizers.l2(weight_decay))(y_a)
        y_a = BatchNormalization()(y_a)
        y_a = Activation('relu')(y_a)
        y_a = MaxPooling2D(pool_size=pool_size_a_1, strides=2)(y_a)

    # CONV BLOCK 2
    filt_size_a_2 = (3, 3)
    pool_size_a_2 = (2, 2)

    if include_layers[2]:
        y_a = Conv2D(num_filters[2], filt_size_a_2, padding='same',
                     kernel_initializer='he_normal',
                     name='conv_3',
                     kernel_regularizer=regularizers.l2(weight_decay))(y_a)
        y_a = BatchNormalization()(y_a)
        y_a = Activation('relu')(y_a)

    if include_layers[3]:
        y_a = Conv2D(num_filters[3], filt_size_a_2, padding='same',
                     kernel_initializer='he_normal',
                     name='conv_4',
                     kernel_regularizer=regularizers.l2(weight_decay))(y_a)
        y_a = BatchNormalization()(y_a)
        y_a = Activation('relu')(y_a)
        y_a = MaxPooling2D(pool_size=pool_size_a_2, strides=2)(y_a)

    # CONV BLOCK 3
    filt_size_a_3 = (3, 3)
    pool_size_a_3 = (2, 2)

    if include_layers[4]:
        y_a = Conv2D(num_filters[4], filt_size_a_3, padding='same',
                     kernel_initializer='he_normal',
                     name='conv_5',
                     kernel_regularizer=regularizers.l2(weight_decay))(y_a)
        y_a = BatchNormalization()(y_a)
        y_a = Activation('relu')(y_a)
    
    if include_layers[5]:
        y_a = Conv2D(num_filters[5], filt_size_a_3, padding='same',
                     kernel_initializer='he_normal',
                     name='conv_6',
                     kernel_regularizer=regularizers.l2(weight_decay))(y_a)
        y_a = BatchNormalization()(y_a)
        y_a = Activation('relu')(y_a)
        y_a = MaxPooling2D(pool_size=pool_size_a_3, strides=2)(y_a)

    # CONV BLOCK 4
    filt_size_a_4 = (3, 3)
    pool_size_a_4 = (32, 24)
    if include_layers[6]:
        y_a = Conv2D(num_filters[6], filt_size_a_4, padding='same',
                     kernel_initializer='he_normal',
                     name='conv_7',
                     kernel_regularizer=regularizers.l2(weight_decay))(y_a)
        y_a = BatchNormalization()(y_a)
        y_a = Activation('relu')(y_a)
    
    if include_layers[7]:
        y_a = Conv2D(num_filters[7], filt_size_a_4,
                     kernel_initializer='he_normal',
                     name='conv_8', padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    
        y_a = BatchNormalization()(y_a)
        y_a = Activation('relu')(y_a)
        y_a = MaxPooling2D(pool_size=pool_size_a_4)(y_a)

    y_a = Flatten()(y_a)

    m = Model(inputs=x_a, outputs=y_a)
    m.name = 'student_model'

    return m, x_a, y_a


def load_audio_model_for_pruning(weight_path, model_type = 'cnn_L3_melspec2'):
    
    m, inputs, outputs = load_model(weight_path, model_type, return_io=True, src_num_gpus=1)

    audio_model = m.get_layer('audio_model')
    count = 1

    for layer in audio_model.layers:
        layer_name = layer.name
        
        if (layer_name[0:6] == 'conv2d' or layer_name == 'audio_embedding_layer'):
            #Rename the conv layers as conv_1, conv_2 .... conv_8 
            audio_model.get_layer(name=layer.name).name='conv_'+str(count)
            count += 1
            #print (layer.name)
    return m, audio_model


def test(model, validation_data_dir, learning_rate=1e-4, validation_epoch_size=1024, validation_batch_size=64, random_state=20180216):
    loss = 'binary_crossentropy'
    metrics = ['accuracy']

    model.compile(Adam(lr=learning_rate),
                  loss=loss,
                  metrics=metrics)

    val_gen = single_epoch_data_generator(validation_data_dir,
                                          validation_epoch_size,
                                          batch_size=validation_batch_size,
                                          kd_model=False,
                                          random_state=random_state)

    val_gen = pescador.maps.keras_tuples(val_gen,
                                         ['video', 'audio'],
                                         'label')
    score = model.evaluate_generator(val_gen, steps=validation_epoch_size)
    
    return score


def get_restart_info(history_path):
    last = None
    with open(history_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row

    return int(last['epoch']), float(last['val_acc']), float(last['val_loss'])


def train(train_data_dir, validation_data_dir, model_to_train = None, include_layers = [1, 1, 1, 1, 1, 1, 1, 1],
          num_filters = [64, 128, 256, 512], pruning=True, finetune=False, output_dir = None, num_epochs=300,
          train_epoch_size=4096, validation_epoch_size=1024, train_batch_size=64, validation_batch_size=64,
          model_type = 'cnn_L3_melspec2', log_path=None, disable_logging=False, random_state=20180216,
          learning_rate=0.001, verbose=True, checkpoint_interval=10, gpus=1, sparsity=[], continue_model_dir=None,
          gsheet_id=None, google_dev_app_name=None):

    init_console_logger(LOGGER, verbose=verbose)
    if not disable_logging:
        init_file_logger(LOGGER, log_path=log_path)
    LOGGER.debug('Initialized logging.')

    kd_flag = False


    if pruning and finetune:
        if output_dir is None:
            output_dir = '/scratch/sk7898/pruning_finetune_output'
        model_attribute = 'pruning_finetune'
    elif pruning and not finetune:
        if output_dir is None:
            output_dir = '/scratch/sk7898/pruning_kd_output'
        kd_flag = True
        model_attribute = 'pruning_kd'
    else:
        if output_dir is None:
            output_dir = '/scratch/sk7898/reduced_kd_output'
        kd_flag = True
        model_attribute = 'reduced_kd'

    # Form model ID
    data_subset_name = os.path.basename(train_data_dir)
    data_subset_name = data_subset_name[:data_subset_name.rindex('_')]
    model_id = os.path.join(data_subset_name, model_type)

    param_dict = {
        'username': getpass.getuser(),
        'train_data_dir': train_data_dir,
        'validation_data_dir': validation_data_dir,
        'model_id': model_id,
        'output_dir': output_dir,
        'include_layers': include_layers,
        'num_filters': num_filters,
        'sparsity': sparsity,
        'pruning': pruning,
        'finetune': finetune,
        'knowledge_distilled': kd_flag,
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

    # Make sure the directories we need exist
    if continue_model_dir:
        model_dir = continue_model_dir
    else:
        model_dir = os.path.join(output_dir, 'embedding', model_attribute, model_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        
    if continue_model_dir:
        latest_model_path = os.path.join(continue_model_dir, 'model_latest.h5')
        model = load_model(latest_model_path)
    else:
        if pruning:
            model = model_to_train
        else:
            model, x_a, y_a = load_student_audio_model_withFFT(include_layers = include_layers,\
                                                               num_filters = num_filters)
    

    param_dict['model_dir'] = model_dir
    train_config_path = os.path.join(model_dir, 'config.json')
    with open(train_config_path, 'w') as fd:
        json.dump(param_dict, fd, indent=2)

    LOGGER.info('Compiling model...')
    if finetune:
        model.compile(Adam(lr=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile(Adam(lr=learning_rate),
                      loss='mean_squared_error',
                      metrics=['mae'])

    LOGGER.info('Model files can be found in "{}"'.format(model_dir))

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

    # Save the model
    '''
    model_json_path = os.path.join(model_dir, 'model.json')
    model_json = model.to_json()
    with open(model_json_path, 'w') as fd:
        json.dump(model_json, fd, indent=2)
    '''

    latest_weight_path = os.path.join(model_dir, 'model_latest.h5')
    best_valid_acc_weight_path = os.path.join(model_dir, 'model_best_valid_accuracy.h5')
    best_valid_loss_weight_path = os.path.join(model_dir, 'model_best_valid_loss.h5')
    checkpoint_weight_path = os.path.join(model_dir, 'model_checkpoint.{epoch:02d}.h5')

    # Load information about last epoch for initializing callbacks and data generators
    if continue_model_dir is not None:
        prev_train_hist_path = os.path.join(continue_model_dir, 'history_csvlog.csv')
        last_epoch_idx, last_val_acc, last_val_loss = get_restart_info(prev_train_hist_path)

    # Set up callbacks
    cb = []
    cb.append(keras.callbacks.ModelCheckpoint(latest_weight_path,
                                              save_weights_only=False,
                                              verbose=1))

    # Accuracy only relevant when no knowledge distillation
    if not kd_flag:
        best_val_acc_cb = keras.callbacks.ModelCheckpoint(best_valid_acc_weight_path,
                                                          save_weights_only=True,
                                                          save_best_only=True,
                                                          verbose=1,
                                                          monitor='val_acc')
        if continue_model_dir is not None:
            best_val_acc_cb.best = last_val_acc
        cb.append(best_val_acc_cb)

    best_val_loss_cb = keras.callbacks.ModelCheckpoint(best_valid_loss_weight_path,
                                                       save_weights_only=False,
                                                       save_best_only=True,
                                                       verbose=1,
                                                       monitor='val_loss')
    if continue_model_dir is not None:
        best_val_loss_cb.best = last_val_loss
    cb.append(best_val_loss_cb)

    checkpoint_cb = keras.callbacks.ModelCheckpoint(checkpoint_weight_path,
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
        cb.append(GSheetLogger(google_dev_app_name, gsheet_id, param_dict))

    LOGGER.info('Setting up train data generator...')
    if continue_model_dir is not None:
        train_start_batch_idx = train_epoch_size * (last_epoch_idx + 1)
    else:
        train_start_batch_idx = None

    if finetune:
        train_gen = data_generator(train_data_dir,
                                   kd_model=False,
                                   batch_size=train_batch_size,
                                   random_state=random_state,
                                   start_batch_idx=train_start_batch_idx)

        train_gen = pescador.maps.keras_tuples(train_gen,
                                               ['video', 'audio'],
                                               'label')

        val_gen = single_epoch_data_generator(validation_data_dir,
                                              validation_epoch_size,
                                              kd_model=False,
                                              batch_size=validation_batch_size,
                                              random_state=random_state)

        val_gen = pescador.maps.keras_tuples(val_gen,
                                             ['video', 'audio'],
                                             'label')

    else:
        train_gen = data_generator(train_data_dir,
                                   kd_model=True,
                                   batch_size=train_batch_size,
                                   random_state=random_state,
                                   start_batch_idx=train_start_batch_idx)

        train_gen = pescador.maps.keras_tuples(train_gen,
                                               'audio',
                                               'label')

        val_gen = single_epoch_data_generator(validation_data_dir,
                                              validation_epoch_size,
                                              kd_model=True,
                                              batch_size=validation_batch_size,
                                              random_state=random_state)

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
    return history


def initialize_weights(masked_model, sparse_model, is_L3=True):
    if is_L3:
        masked_model.set_weights(sparse_model.get_weights())
    else:
        masked_model.set_weights(sparse_model.get_layer('audio_model').get_weights())
    
    return masked_model


def retrain(l3_model, masks, train_data_dir, validation_data_dir, finetune=True, **kwargs):
    if finetune:
        l3_model_kd, x_a, y_a = construct_cnn_L3_melspec2_kd(masks)
        model = initialize_weights(l3_model_kd, l3_model, is_L3=True)  
        train(train_data_dir, validation_data_dir, model, pruning=True, finetune=finetune, **kwargs)
    else:
        audio_model, x_a, y_a = construct_cnn_L3_melspec2_kd_audio_model(masks)
        audio_model = initialize_weights(audio_model, l3_model, is_L3=False)
        train(train_data_dir, validation_data_dir, audio_model, pruning=True, finetune=finetune, **kwargs)


def zero_check(sparsity_list):
    for val in sparsity_list:
        if val:
            return False
    return True


def printList(plist):
    printStr = "List Values: "
    for val in plist:
        printStr += str(val) + ", "
    print(printStr)

def printDict(pDict):
    printStr = "Dictionary Key and Values: "
    for key in pDict:
        printStr += str(key)+ ": "+ str(pDict[key])+ ", "
    print(printStr)

def drop_filters(model, new_model, sparsity_dict):
    weights_dict = {}
    for i, layer in enumerate(model.layers):
        if layer.name in sparsity_dict:
            if sparsity_dict[layer.name]:
                filters_mag = []
                new_weights = np.empty_like(new_model.get_layer(layer.name).get_weights())

                if layer.name in weights_dict:
                    weights = weights_dict[layer.name]
                else:
                    weights = layer.get_weights()[0]
            
                biases = layer.get_weights()[1]
            
                for channel in range(weights.shape[3]):
                    filters_mag.append(np.sum(np.abs(weights[:, :, :, channel]))) 
            
                threshold = calculate_threshold(filters_mag, sparsity_dict[layer.name])    
                mask      = K.cast(K.greater(filters_mag, threshold), dtypes.float32)
            
                #print(K.eval(threshold))
                non_dropped_ch = K.eval(tf.where(mask > 0))
            
                new_weights[0] = K.eval(tf.gather(weights, [y for x in non_dropped_ch for y in x], axis = -1))
                new_weights[1] = K.eval(tf.gather(biases, [y for x in non_dropped_ch for y in x], axis = 0))

                next_conv = 'conv_'+ str(int(layer.name[5])+1)
                if not 'conv_8' in layer.name: 
                    weights_dict[next_conv] = K.eval(tf.gather(model.get_layer(next_conv).get_weights()[0], \
                                                               [y for x in non_dropped_ch for y in x], axis = 2))
                
                new_model.get_layer(layer.name).set_weights(new_weights)
                #print(new_weights[0].shape)
            
            else:
                new_model.get_layer(layer.name).set_weights(layer.get_weights())
                #print(layer.get_weights()[0].shape)

    return new_model
            

def get_sparsity_filters(conv_layers, conv_filters, sparsity_filters):
    sparsity_dict = {}
    new_num_filters = []
    for i, layer in enumerate(conv_layers):
        if sparsity_filters[i]:
            new_num_filters.append(int((100. - sparsity_filters[i])/100 * conv_filters[i]))
        else:
            new_num_filters.append(conv_filters[i])

        sparsity_dict[layer] = sparsity_filters[i]

    return sparsity_dict, new_num_filters


def pruning(weight_path, train_data_dir, validation_data_dir, output_dir = '/scratch/sk7898/pruned_model',
            blockwise=False, layerwise=False, filterwise=False, per_layer=False, sparsity=[], test_model=False, save_model=False,
            retrain_model=False, finetune = True, **kwargs):
    
    conv_blocks = 4
    
    sparsity_values = [70., 80., 85., 90., 95.]
    sparsity_blks = [0, 0, 0, 0]
    if per_layer:
        sparsity_layers = [0, 0, 0, 0, 0, 0, 0, 0]

    elif not per_layer and not filterwise:
        if sparsity==[]:
            sparsity_layers = [[0, 60., 60., 70., 50., 70., 70., 80.],
                               [0, 70., 70., 75., 60., 80., 80., 85.],
                               [0, 80., 80., 85., 40., 85., 85., 95.]]
        else:
            sparsity_layers = [sparsity]

    elif filterwise:
        #Use values like 0.5, 0.625, 0.75, 0.875, 0.9375
        sparsity_filters = [0, 50., 50., 50., 50., 50., 50., 50.]
        filter_sparsity = {}
        conv_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8']
        conv_filters = [64, 64, 128, 128, 256, 256, 512, 512]
        model, audio_model = load_audio_model_for_pruning(weight_path)
        filter_sparsity, num_filters = get_sparsity_filters(conv_layers, conv_filters, sparsity_filters)
        
        new_model, x_a, y_a = load_student_audio_model_withFFT(include_layers = [1, 1, 1, 1, 1, 1, 1, 1],\
                                                     num_filters = num_filters)
        reduced_model = drop_filters(audio_model, new_model, filter_sparsity) 

    else:
        print("Incorrect Pruning selection")


    if(blockwise and zero_check(sparsity_blks)):
        LOGGER.info("Block-wise Pruning")
        LOGGER.info("*********************************************************************")
    
        for sparsity in sparsity_values:
            for blockid in range(conv_blocks):
                model, audio_model = load_audio_model_for_pruning(weight_path)
                sparsity_vals = get_sparsity_blocks(blockid, sparsity, sparsity_blks) 
                sparsified_model = sparsify_block(audio_model, sparsity_vals)
            
                model.get_layer('audio_model').set_weights(sparsified_model.get_weights()) 
            
                score = test(model, validation_data_dir)
                LOGGER.info('Conv Block Pruned: {0} Sparsity Value: {1}'.format(blockid+1, sparsity))
                LOGGER.info('Loss: {0} Accuracy: {1}'.format(score[0], score[1]))
                LOGGER.info('---------------------------------------------------------------')
    
    if(layerwise and per_layer):
        LOGGER.info("Layer-wise Pruning")
        LOGGER.info("**********************************************************************")
        for sparsity in sparsity_values:
            for layerid in range(conv_blocks*2):
                model, audio_model = load_audio_model_for_pruning(weight_path)
                sparsity_vals = get_sparsity_layers(layerid, sparsity, sparsity_layers)
                sparsified_model, masks = sparsify_layer(audio_model, sparsity_vals)

                model.get_layer('audio_model').set_weights(sparsified_model.get_weights())

                score = test(model, validation_data_dir)
                LOGGER.info('Conv Layer Pruned: {0} Sparsity Value: {1}'.format(layerid+1, sparsity))
                LOGGER.info('Loss: {0} Accuracy: {1}'.format(score[0], score[1]))
                LOGGER.info('----------------------------------------------------------------')
    
    if(layerwise and not per_layer):
        LOGGER.info("Specific Pruning Values")
        LOGGER.info("**********************************************************************")
        for sparsity in sparsity_layers:
            LOGGER.info('Sparsity: {0}'.format(sparsity))
            model, audio_model = load_audio_model_for_pruning(weight_path)
            sparsity_vals = get_sparsity_layers(None, None, sparsity)
            sparsified_model, masks = sparsify_layer(audio_model, sparsity_vals)
            
            model.get_layer('audio_model').set_weights(sparsified_model.get_weights())
            if test_model:
                score = test(model, validation_data_dir)
                printList(sparsity)
                LOGGER.info('Loss: {0} Accuracy: {1}'.format(score[0], score[1]))
                LOGGER.info('----------------------------------------------------------------')
            
            if save_model:
                pruned_model_name = 'pruned_audio_'+str(score[1])+'.h5'
                pruned_model_path = os.path.join(output_dir, pruned_model_name)
                sparsified_model.save(pruned_model_path)

            if retrain_model:
                if(finetune):
                    LOGGER.info('Retraining model with fine tuning')
                else:
                    LOGGER.info('Retraining model with knowledge distillation')
                    retrain(model, masks, train_data_dir, validation_data_dir, sparsity=sparsity,
                            finetune = finetune, **kwargs)

def main():
    is_pruning = True
    train_data_dir = '/beegfs/work/AudioSetSamples/music_train'
    validation_data_dir = '/beegfs/work/AudioSetSamples/music_valid'

    if is_pruning:
        pruning(weight_path, train_data_dir, validation_data_dir, save_model=True,
                test_model=True, retrain_model=False, finetune=False)
    else:
        include_layers = [1, 1, 1, 1, 1, 1, 1, 1]
        num_filters = [64, 64, 128, 128, 256, 256]
        train(train_data_dir, validation_data_dir, include_layers = include_layers,
              num_filters = num_filters, pruning=False, finetune=False, continue_model_dir=None)

if __name__=='__main__':
    main()

