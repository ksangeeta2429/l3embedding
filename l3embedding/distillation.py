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
from keras import backend as K
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model
from keras.losses import categorical_crossentropy
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

LOGGER = logging.getLogger('l3distillation')
LOGGER.setLevel(logging.DEBUG)

nb_classes = 2

class MultiGPUCheckpointCallback(Callback):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):

        super(MultiGPU_Checkpoint_Callback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)


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

    def __init__(self, google_dev_app_name, spreadsheet_id, param_dict, loss_type):
        super(GSheetLogger).__init__()
        self.loss_type = loss_type
        self.google_dev_app_name = google_dev_app_name
        self.spreadsheet_id = spreadsheet_id
        self.credentials = get_credentials(google_dev_app_name)
        self.service = discovery.build('sheets', 'v4', credentials=self.credentials)
        self.param_dict = copy.deepcopy(param_dict)

        row_num = get_row(self.service, self.spreadsheet_id, self.param_dict, 'distillation')
        if row_num is None:
            append_row(self.service, self.spreadsheet_id, self.param_dict, 'distillation')

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
        if self.loss_type != 'mse':
            if latest_train_acc > self.best_train_acc:
                self.best_train_acc = latest_train_acc
            if latest_valid_acc > self.best_valid_acc:
                self.best_valid_acc = latest_valid_acc

        values = [latest_epoch, latest_train_loss, latest_valid_loss,
                  latest_train_acc, latest_valid_acc, self.best_train_loss,
                  self.best_valid_loss, self.best_train_acc, self.best_valid_acc]

        update_experiment(self.service, self.spreadsheet_id, self.param_dict,
                          'X', 'AF', values, 'distillation')


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



def load_student_model(student, temp=5, return_io=True):
    student = model.load_weights(weight_path)

    # Remove the softmax layer from the student network
    student.layers.pop()

    # Now collect the logits from the last layer
    logits = student.layers[-1].output 
    probs = Activation('softmax')(logits)

    # soft probabilities at raised temperature
    logits_T = Lambda(lambda x: x / temp)(logits)
    probs_T = Activation('softmax')(logits_T)

    output = concatenate([probs, probs_T])

    # This is our new student model
    student = Model(student.input, output)

    if return_io:
        return student, student.input, output
    else:
        return student


def get_teacher_logits(teacher, video_batch, audio_batch, temp=5):
    if teacher is None:
        print("Teacher L3 not provided. Exiting!")
        exit(0)
    
    try:
        with tf.Graph().as_default():    
            #Remove softmax from l3
            teacher.layers.pop()
            logits = teacher.layers[-1].output
            
            # soft probabilities
            logits_T = Lambda(lambda x: x/temp)(logits)
            probabilities_T = Activation('softmax')(logits_T)
            
            #Collect the logits from the previous layer output and store it in a different model
            teacher_WO_Softmax = Model(teacher.input, probabilities_T)
            teacher_logits = teacher_WO_Softmax.predict([video_batch, audio_batch])
            return teacher_logits

    except GeneratorExit:
        pass


def get_teacher_embeddings(teacher):
    if teacher is None:
        print("Teacher L3 not provided. Exiting!")
        exit(0)

    try:
        with tf.Graph().as_default:
            audio_model = teacher.get_layer('audio_model')
            embed_layer = audio_model.get_layer('audio_embedding_layer')
            y_a = MaxPooling2D(pool_size=pool_size, padding='same')(embed_layer.output)
            y_a = Flatten()(y_a)
            teacher_audio = Model(inputs=audio_model.input, outputs=y_a)
            embeddings = teacher_audio.predict(batch['audio'])
            return embeddings

    except GeneratorExit:
        pass

def data_generator(data_dir, teacher, loss_type='entropy', temp=5, batch_size=512, random_state=20180216, start_batch_idx=None, keys=None):
    random.seed(random_state)

    batch = None
    curr_batch_size = 0
    batch_idx = 0
    file_idx = 0
    start_label_idx = 0

    # Limit keys to avoid producing batches with all of the metadata fields
    if not keys:
        if loss_type == 'mse':
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
                # If we are starting from a particular batch, skip yielding all of the prior batches
                if start_batch_idx is None or batch_idx >= start_batch_idx:
                    # Convert audio to float
                    batch['audio'] = pcm2float(batch['audio'], dtype='float32')

                    if loss_type == 'entropy':
                        # Preprocess video so samples are in [-1,1]
                        batch['video'] = 2 * img_as_float(batch['video']).astype('float32') - 1
                        
                        teacher_logits = get_teacher_logits(teacher, batch['video'], batch['audio'], temp)
                        batch['labels'] = np.concatenate([batch['labels'], teacher_logits], axis=1)
                        
                    if loss_type == 'mse':
                        # Get the embedding layer output from the audio_model and flatten it to be treated as labels for the student audio model
                        batch['label'] = get_teacher_embeddings(teacher)                            
                                                
                    yield batch

                batch_idx += 1
                curr_batch_size = 0
                batch = None


def single_epoch_data_generator(data_dir, epoch_size, teacher, **kwargs):
    while True:
        data_gen = data_generator(data_dir, teacher, **kwargs)
        for idx, item in enumerate(data_gen):
            yield item
            # Once we generate all batches for an epoch, restart the generator
            if (idx + 1) == epoch_size:
                break


def knowledge_distillation_loss(y_true, y_pred, lambda_constant):
    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    y_true, y_true_softs = y_true[: , :nb_classes], y_true[: , nb_classes:]
    y_pred, y_pred_softs = y_pred[: , :nb_classes], y_pred[: , nb_classes:]
    loss = categorical_crossentropy(y_true, y_pred) + lambda_constant * categorical_crossentropy(y_true_softs, y_pred_softs)
    return loss


def kd_accuracy(y_true, y_pred):
    # For testing use regular output probabilities - without temperature
    y_true = y_true[:, :nb_classes]
    y_pred = y_pred[:, :nb_classes]
    return categorical_accuracy(y_true, y_pred)


def train(train_data_dir, validation_data_dir, student_weight_path, teacher_weight_path, output_dir = None, loss_type='entropy',\
          num_epochs=300, train_epoch_size=4096, validation_epoch_size=1024, train_batch_size=64, validation_batch_size=64,\
          model_type = 'cnn_L3_melspec2', log_path=None, disable_logging=False, random_state=20180216, temp=5, lambda_constant=0.8,\
          learning_rate=0.00001, verbose=True, checkpoint_interval=10, gpus=1, continue_model_dir=None,\
          gsheet_id=None, google_dev_app_name=None, thresholds=None):

    init_console_logger(LOGGER, verbose=verbose)
    if not disable_logging:
        init_file_logger(LOGGER, log_path=log_path)
    LOGGER.debug('Initialized logging.')
    LOGGER.info('Student Model: %s', student_weight_path)
    LOGGER.info('Teaacher Model: %s', teacher_weight_path)

    
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
        'student': student_weight_path,
        'teacher': teacher_weight_path,
        'loss_type': loss_type,
        'temperature': temp,
        'lambda': lambda_constant,
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

    teacher = keras.models.load_model(teacher_weight_path, custom_objects={'Melspectrogram': Melspectrogram})

    # Make sure the directories we need exist
    if continue_model_dir:
        model_dir = continue_model_dir
    else:
        model_dir = os.path.join(output_dir, 'embedding', model_attribute, model_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        
    if continue_model_dir:
        latest_model_path = os.path.join(continue_model_dir, 'model_latest.h5')
        student_base_model, inputs, outputs  = keras.models.load_model(latest_model_path, custom_objects={'Melspectrogram': Melspectrogram})
    else:
        student = keras.models.load_model(student_weight_path, custom_objects={'Melspectrogram': Melspectrogram})
        student_base_model, inputs, outputs  = load_student_model(student, student_weight_path, temp=temp, return_io=True)
        
    model = multi_gpu_model(student_base_model, gpus=gpus)

    param_dict['model_dir'] = model_dir
    train_config_path = os.path.join(model_dir, 'config.json')
    with open(train_config_path, 'w') as fd:
        json.dump(param_dict, fd, indent=2)

    LOGGER.info('Compiling model...')
    if loss_type == 'entropy':
        model.compile(optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True), #Adam(lr=learning_rate)
                      loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, lambda_constant),
                      metrics=[kd_accuracy])
    
    elif loss_type == 'mse':
        model.compile(Adam(lr=learning_rate),
                      loss='mean_squared_error',
                      metrics=['mae'])
    else:
        raise ValueError('Invalid loss type: "{}"'.format(loss_type))

    
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
    cb.append(MultiGPUCheckpointCallback(latest_weight_path,
                                         student_base_model,
                                         save_weights_only=False,
                                         verbose=1))

    if loss_type == 'entropy':
        best_val_acc_cb = MultiGPUCheckpointCallback(best_valid_acc_weight_path,
                                                     student_base_model,
                                                     save_weights_only=False,
                                                     save_best_only=True,
                                                     verbose=1,
                                                     monitor='val_acc')
        if continue_model_dir is not None:
            best_val_acc_cb.best = last_val_acc
        cb.append(best_val_acc_cb)

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
        cb.append(GSheetLogger(google_dev_app_name, gsheet_id, param_dict, kd_flag))

    LOGGER.info('Setting up train data generator...')
    if continue_model_dir is not None:
        train_start_batch_idx = train_epoch_size * (last_epoch_idx + 1)
    else:
        train_start_batch_idx = None


    train_gen = data_generator(train_data_dir,
                               teacher,
                               loss_type=loss_type,
                               temp=temp,
                               batch_size=train_batch_size,
                               random_state=random_state,
                               start_batch_idx=train_start_batch_idx)

    val_gen = single_epoch_data_generator(validation_data_dir,
                                          validation_epoch_size,
                                          teacher,
                                          loss_type=loss_type,
                                          temp=temp,
                                          batch_size=validation_batch_size,
                                          random_state=random_state)


    if loss_type == 'entropy':
        
        train_gen = pescador.maps.keras_tuples(train_gen,
                                               ['video', 'audio'],
                                               'label')
        
        val_gen = pescador.maps.keras_tuples(val_gen,
                                             ['video', 'audio'],
                                             'label')

    elif loss_type == 'mse':
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
