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
from .training_utils import conv_dict_to_val_list, multi_gpu_model
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
from kapre.time_frequency import Spectrogram, Melspectrogram

LOGGER = logging.getLogger('prunedl3embedding')
LOGGER.setLevel(logging.DEBUG)
CONV_LAYERS = 8

##########
# Pruning Layers:
# Step 1: Prune each layer with different aprsity levels to decide the sensitivity of each 
#         Prune whole model after deciding the sensitivity of each layer and test the performance
# Step 2: Fine-tune the whole model at a go
# Step 3: Use knowledge distillation setting to retrain the pruned model. This does not take into account the AVC problem.


##########
# Pruning Filters (whole feature-maps)
# Pseudo-code
# Step 1: Get the magnitude of each filter
# Step 2: Experiment with different sparsity values and drop the resulting insignificant filters (with magnitude lesser than the threshold)
# Step 3: Drop the channels corresponding to the dropped filters in the layer's output
# Step 4: Form the new architecture according to the new number of filters
# Step 5: Freeze the video model and fine-tune the audio and the merged Dense layers
##########

graph = tf.get_default_graph()
audio_model = None

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

    def __init__(self, google_dev_app_name, spreadsheet_id, param_dict, is_kd):
        super(GSheetLogger).__init__()
        self.is_kd = is_kd
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
        if not self.is_kd:
            if latest_train_acc > self.best_train_acc:
                self.best_train_acc = latest_train_acc
            if latest_valid_acc > self.best_valid_acc:
                self.best_valid_acc = latest_valid_acc

        values = [
            latest_epoch, latest_train_loss, latest_valid_loss,
            latest_train_acc, latest_valid_acc, self.best_train_loss,
            self.best_valid_loss, self.best_train_acc, self.best_valid_acc]

        update_experiment(self.service, self.spreadsheet_id, self.param_dict,
                          'X', 'AF', values, 'prunedembedding')


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


def data_generator(data_dir, weight_path, kd_model=False, batch_size=512, random_state=20180216, start_batch_idx=None, keys=None):
    random.seed(random_state)

    batch = None
    curr_batch_size = 0
    batch_idx = 0
    file_idx = 0
    start_label_idx = 0
    audio_model = load_embedding(weight_path, model_type = 'cnn_L3_melspec2', embedding_type = 'audio', \
                                 pooling_type = 'short', kd_model=False, tgt_num_gpus = 1)

    global graph

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


def single_epoch_data_generator(data_dir, weight_path, epoch_size, kd_model, **kwargs):
    while True:
        data_gen = data_generator(data_dir, weight_path, kd_model, **kwargs)
        for idx, item in enumerate(data_gen):
            yield item
            # Once we generate all batches for an epoch, restart the generator
            if (idx + 1) == epoch_size:
                break


def get_sparsity_layers(layer, sparsity_layer, sparsity_value_layers):
    sparsity = {}
    for idx in range(len(sparsity_value_layers)):
        if idx+1 == 8:
            layer_name = 'audio_embedding_layer'
        else:
            layer_name = 'conv_'+str(idx+1)
        if(layer and sparsity_layer and idx == layer):
            sparsity[layer_name] = sparsity_layer
        else:
            sparsity[layer_name] = sparsity_value_layers[idx]

    return sparsity


def calculate_threshold(weights, ratio):
    np_thresh = np.percentile(np.abs(weights), ratio)
    tf_thresh = tf.contrib.distributions.percentile(K.abs(weights), ratio)
    return tf_thresh, np_thresh

        
def sparsify_layer(model, sparsity_dict):
    masks = {}
    thresholds = {}
    for layer in model.layers:
        if 'conv_' in layer.name or 'audio_embedding' in layer.name:
            target_weights = np.empty_like(layer.get_weights())
            weights = layer.get_weights()[0]

            if sparsity_dict[layer.name]:
                threshold, np_threshold = calculate_threshold(weights, sparsity_dict[layer.name])
                tf_threshold = K.eval(threshold)
                mask      = K.cast(K.greater(K.abs(weights), threshold), dtypes.float32)
                new_weights = weights * K.eval(mask)
                target_weights[0] = new_weights
            
                target_weights[1] = layer.get_weights()[1]
                layer.set_weights(target_weights)
            
            else:
                tf_threshold = 0.0
                mask = K.ones_like(weights)

            masks[layer.name] = mask
            thresholds[layer.name] = tf_threshold

    return model, masks, thresholds


def load_audio_model_for_pruning(weight_path, model_type = 'cnn_L3_melspec2'):
    
    m, inputs, outputs = load_model(weight_path, model_type, return_io=True, src_num_gpus=1)

    count = 1

    for layer in audio_model.layers:
        layer_name = layer.name
        
        if (layer_name[0:6] == 'conv2d':
            audio_model.get_layer(name=layer.name).name='conv_'+str(count)
            count += 1
    return m, audio_model


def test(model, weight_path, validation_data_dir, learning_rate=1e-4, validation_epoch_size=1024, validation_batch_size=64, random_state=20180216):
    loss = 'binary_crossentropy'
    metrics = ['accuracy']

    model.compile(Adam(lr=learning_rate),
                  loss=loss,
                  metrics=metrics)

    val_gen = single_epoch_data_generator(validation_data_dir,
                                          weight_path,
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


def gpu_wrapper_4gpus(model_f):
    """
    Decorator for creating multi-gpu models
    """
    def wrapped(*args, **kwargs):
        m, inp, out = model_f(*args, **kwargs)
        
        num_gpus = 4
        if num_gpus > 1:
            m = keras.utils.multi_gpu_model(m, gpus=num_gpus)
            m.save_weights('after_multi.h5')

        return m, inp, out

    return wrapped


@gpu_wrapper_4gpus
def initialize_weights(masked_model=None, sparse_model=None, is_L3=True, input=None, output=None):
    embedding_length_original = sparse_model.get_layer('audio_model').output_shape
    if is_L3:
        embedding_length_new = masked_model.get_layer('audio_model').output_shape

        if embedding_length_new != embedding_length_original or len(sparse_model.get_layer('audio_model').layers) < 8:
            LOGGER.info("New embedding Length: {0}".format(embedding_length_new))
            new_video_model = masked_model.get_layer('vision_model')
            old_video_model = sparse_model.get_layer('vision_model')
            new_audio_model = masked_model.get_layer('audio_model')
            old_audio_model = sparse_model.get_layer('audio_model')

            new_video_model.set_weights(old_video_model.get_weights())
            
        else:
            masked_model.set_weights(sparse_model.get_weights())
            
        
        for layer in masked_model.get_layer('vision_model').layers:
            layer.trainable = False
    else:
        masked_model.set_weights(sparse_model.get_layer('audio_model').get_weights())
        
    return masked_model, input, output


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

def drop_filters(model, new_model, new_filters_dict, old_filters_dict):
    weights_dict = {}
    for i, layer in enumerate(model.layers):
        if layer.name in new_filters_dict:
            if new_filters_dict[layer.name] != old_filters_dict[layer.name]:
                filters_mag = []
                sorted_idx = []
                num_filters_drop = old_filters_dict[layer.name] - new_filters_dict[layer.name]
                new_weights = np.empty_like(new_model.get_layer(layer.name).get_weights())

                if layer.name in weights_dict:
                    weights = weights_dict[layer.name]
                else:
                    weights = layer.get_weights()[0]
            
                biases = layer.get_weights()[1]
            
                for channel in range(weights.shape[3]):
                    filters_mag.append(np.sum(np.abs(weights[:, :, :, channel]))) 
            
                sorted_idx = np.argsort(np.array(filters_mag))
                sorted_idx = sorted_idx[num_filters_drop:]
            
                new_weights[0] = K.eval(tf.gather(weights, sorted_idx, axis = -1))
                new_weights[1] = K.eval(tf.gather(biases, sorted_idx, axis = 0))

                if not 'audio_embedding' in layer.name:
			if int(layer.name[5]) != 7:
            			next_conv = 'conv_'+ str(int(layer.name[5])+1)
			else:
				next_conv = 'audio_embedding_layer'
            		
			weights_dict[next_conv] = K.eval(tf.gather(model.get_layer(next_conv).get_weights()[0], \
                                                               sorted_idx, axis = 2))
                
                new_model.get_layer(layer.name).set_weights(new_weights)
            
            else:
                if layer.name not in weights_dict:
                    new_model.get_layer(layer.name).set_weights(layer.get_weights())

    return new_model
            

def get_filters(conv_layers, conv_filters, num_filters):
    new_filters = {}
    old_filters = {}
    for i, layer in enumerate(conv_layers):
        old_filters[layer] = conv_filters[i]
        new_filters[layer] = num_filters[i]

    return new_filters, old_filters


def train(train_data_dir, validation_data_dir, weight_path, new_l3 = None, old_l3 = None, include_layers = [1, 1, 1, 1, 1, 1, 1, 1],\
          num_filters = [64, 64, 128, 128, 256, 256, 512, 512], pruning=True, finetune=False, layerwise=False, filterwise=False,\
          output_dir = None, num_epochs=300, train_epoch_size=4096, validation_epoch_size=1024, train_batch_size=64, \
          validation_batch_size=64, model_type = 'cnn_L3_melspec2', log_path=None, disable_logging=False, random_state=20180216,\
          learning_rate=0.0001, verbose=True, checkpoint_interval=10, gpus=1, sparsity=[], \
          gsheet_id=None, google_dev_app_name=None, thresholds=None):

    init_console_logger(LOGGER, verbose=verbose)
    if not disable_logging:
        init_file_logger(LOGGER, log_path=log_path)
    LOGGER.debug('Initialized logging.')

    if finetune:
        LOGGER.info('Retraining model with fine tuning')
    else:
        LOGGER.info('Retraining model with knowledge distillation')

    kd_flag = False

    if pruning and finetune:
        if filterwise:
            model_attribute = 'pruning_finetune_filterwise'
        elif layerwise:
            model_attribute = 'pruning_finetune_layerwise'
        else:
            model_attribute = 'pruning_finetune_reduced'
    elif pruning and not finetune:
        kd_flag = True
        model_attribute = 'pruning_kd'
    else:
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

    if filterwise and not layerwise:
        param_dict['pruning'] = 'filterwise'
    elif layerwise and not filterwise:
        param_dict['pruning'] = 'layerwise'
    elif not filterwise and not layerwise and old_l3 is not None:
        param_dict['pruning'] = 'reduced'
        
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
        if old_l3 is not None:
            model, inputs, outputs  = load_new_model(latest_model_path, new_l3, return_io=True, src_num_gpus=gpus)
        else:
            model, inputs, outputs  = load_new_model(latest_model_path, model_type, return_io=True, src_num_gpus=gpus, thresholds=thresholds)
    else:
        if pruning:
            if finetune:
                if old_l3 is not None: 
                    model, inputs, outputs = initialize_weights(masked_model=new_l3, sparse_model=old_l3, is_L3=True,\
                                                                input=old_l3.inputs, output=old_l3.outputs)
                else:
                    l3_model_kd, x , y  = construct_cnn_L3_melspec2_masked(thresholds=thresholds)
                    model, inputs, outputs = initialize_weights(masked_model=l3_model_kd, sparse_model=new_l3, is_L3=True, input=x , output=y)

            else:
                audio_model, x_a, y_a = construct_cnn_L3_melspec2_masked_audio_model(thresholds)
                model, x_a, y_a = initialize_weights(masked_model=audio_model, sparse_model=new_l3, is_L3=False, input=x_a, output=y_a)           
        else:
            model, x_a, y_a = construct_cnn_L3_melspec2_reduced_audio_model(include_layers = include_layers,\
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
                                              save_weights_only=True,
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
                                                       save_weights_only=True,
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

    history_checkpoint = os.path.join(model_dir, 'history_checkpoint.pkl')
    cb.append(LossHistory(history_checkpoint))

    history_csvlog = os.path.join(model_dir, 'history_csvlog.csv')
    cb.append(keras.callbacks.CSVLogger(history_csvlog, append=True, separator=','))

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10)
    reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

    cb.append(earlyStopping)
    cb.append(reduceLR)

    if gsheet_id:
        cb.append(GSheetLogger(google_dev_app_name, gsheet_id, param_dict, kd_flag))

    LOGGER.info('Setting up train data generator...')
    if continue_model_dir is not None:
        train_start_batch_idx = train_epoch_size * (last_epoch_idx + 1)
    else:
        train_start_batch_idx = None

    if finetune:
        train_gen = data_generator(train_data_dir,
                                   weight_path,
                                   kd_model=False,
                                   batch_size=train_batch_size,
                                   random_state=random_state,
                                   start_batch_idx=train_start_batch_idx)

        train_gen = pescador.maps.keras_tuples(train_gen,
                                               ['video', 'audio'],
                                               'label')

        val_gen = single_epoch_data_generator(validation_data_dir,
                                              weight_path,
                                              validation_epoch_size,
                                              kd_model=False,
                                              batch_size=validation_batch_size,
                                              random_state=random_state)

        val_gen = pescador.maps.keras_tuples(val_gen,
                                             ['video', 'audio'],
                                             'label')

    else:
        train_gen = data_generator(train_data_dir,
                                   weight_path,
                                   kd_model=True,
                                   batch_size=train_batch_size,
                                   random_state=random_state,
                                   start_batch_idx=train_start_batch_idx)

        train_gen = pescador.maps.keras_tuples(train_gen,
                                               'audio',
                                               'label')

        val_gen = single_epoch_data_generator(validation_data_dir,
                                              weight_path,
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


def pruning(weight_path, train_data_dir, validation_data_dir, output_dir,\
            layerwise=True, per_layer=False, filterwise=False, sparsity=[], include_layers=[], num_filters=[],\
            test_model=False, save_model=False, retrain_model=False, finetune = True, **kwargs):
    
    isReduced = False

    if sparsity==[]:
        if per_layer:
            sparsity_list = [0, 0, 0, 0, 0, 0, 0, 0]
        elif filterwise:
            sparsity_list = None
        else:
            sparsity_list = [[0, 60., 60., 70., 50., 70., 70., 80.]]
    else:
        sparsity_list = [sparsity]
        
    if filterwise and num_filters==[]:
        LOGGER.info("ERROR: Please give a list of filters to run filterwise pruning experiment")
        exit(0)
                        
    if len(include_layers) > 0 or len(num_filters) > 0:
        isReduced = True
    
    if layerwise and per_layer:
        LOGGER.info("Layer-wise Pruning")
        LOGGER.info("**********************************************************************")
        
        sparsity_values = [70., 80., 85., 90., 95.]
        for sparsity in sparsity_values:
            for layerid in range(CONV_LAYERS):
                model, audio_model = load_audio_model_for_pruning(weight_path)
                sparsity_vals = get_sparsity_layers(layerid, sparsity, sparsity_list)
                sparsified_model, masks, thresholds = sparsify_layer(audio_model, sparsity_vals)

                model.get_layer('audio_model').set_weights(sparsified_model.get_weights())

                score = test(model, validation_data_dir)
                LOGGER.info('Conv Layer Pruned: {0} Sparsity Value: {1}'.format(layerid+1, sparsity))
                LOGGER.info('Loss: {0} Accuracy: {1}'.format(score[0], score[1]))
                LOGGER.info('----------------------------------------------------------------')
    
    if layerwise and not per_layer:
        LOGGER.info("Sequential Layer Pruning")
        LOGGER.info("**********************************************************************")
        for sparsity in sparsity_list:
            LOGGER.info('Sparsity: {0}'.format(sparsity))
            model, audio_model = load_audio_model_for_pruning(weight_path)
            sparsity_vals = get_sparsity_layers(None, None, sparsity)
            sparsified_model, masks, thresholds = sparsify_layer(audio_model, sparsity_vals)
            print('Thresholds:\n', conv_dict_to_val_list(thresholds))
            
            model.get_layer('audio_model').set_weights(sparsified_model.get_weights())
            
            if test_model:
                score = test(model, validation_data_dir)
                print('Sparsity: ', printList(sparsity))
                print('TEST Loss: ', score[0], '\tAccuracy: ',score[1])
                LOGGER.info('TEST Loss: {0} Accuracy: {1}'.format(score[0], score[1]))
                LOGGER.info('----------------------------------------------------------------')
        
    if isReduced:
        if len(include_layers) == 0:
            include_layers = [1, 1, 1, 1, 1, 1, 1, 1]

        old_model, audio_model = load_audio_model_for_pruning(weight_path)
        new_audio_model, x_a, y_a = construct_cnn_L3_melspec2_reduced_audio_model(include_layers = include_layers,\
                                                                                  num_filters = num_filters)

        if filterwise:
            filter_model_save_str = str(num_filters[0])+"_"+str(num_filters[1])+"_"+str(num_filters[2])+"_"+str(num_filters[3])
            LOGGER.info("Filter Dropping")
            LOGGER.info("**********************************************************************")
        
            conv_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'audio_embedding_layer']
            conv_filters = [64, 64, 128, 128, 256, 256, 512, 512]
 
            LOGGER.info('For filterwise, Sparsity: None')
            new_filters_dict = {}
            old_filters_dict = {}
            new_filters_dict, old_filters_dict = get_filters(conv_layers, conv_filters, num_filters)
            new_audio_model = drop_filters(audio_model, new_audio_model, new_filters_dict, old_filters_dict)
            sparsified_model = new_audio_model 

        vision_model, x_i, y_i = construct_cnn_L3_orig_inputbn_vision_model()
        model, inputs, outputs = L3_merge_audio_vision_models(vision_model, x_i, new_audio_model, x_a, 'cnn_L3_reduced')
        

    if save_model:
        if isReduced:
            new_l3_name = 'reduced_audio_'+filter_model_save_str+'.h5'
        else:
            new_l3_name = 'pruned_audio_'+str(score[1])+'.h5'
        new_model_path = os.path.join(output_dir, new_l3_name)
        sparsified_model.save(new_model_path)

    if retrain_model:
        if isReduced:
            train(train_data_dir, validation_data_dir, weight_path, new_l3=model, old_l3=old_model, sparsity=sparsity, finetune=finetune,\
                  layerwise=layerwise, filterwise=filterwise, output_dir=output_dir, include_layers=include_layers, num_filters=num_filters, **kwargs)
        else:
            train(train_data_dir, validation_data_dir, weight_path, new_l3=model, sparsity=sparsity, thresholds=thresholds,\
                  finetune=finetune, layerwise=layerwise, filterwise=filterwise, output_dir=output_dir, **kwargs)

