import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import getpass
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
import pandas as pd
from keras import backend as K
from keras import activations
from keras.regularizers import l2
from keras.layers import *
from keras.models import Model
from log import *
import oyaml as yaml

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def cycle_shuffle(iterable, shuffle=True):
    lst = list(iterable)
    while True:
        yield from lst
        if shuffle:
            random.shuffle(lst)
            
def get_file_targets(annotation_path, taxonomy_path):
    
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.load(f, Loader=yaml.Loader)
    
    coarse_target_labels = ["_".join([str(k), v])
                            for k, v in taxonomy['coarse'].items()]
         
    file_list = annotation_data.sort_values('audio_filename')['audio_filename'].unique().tolist()
    target_list = {os.path.basename(filename.replace('wav', 'npz')): [] for filename in file_list}
    
    for filename in file_list:
        key = os.path.basename(filename.replace('wav', 'npz'))
        file_df = annotation_data[annotation_data['audio_filename'] == filename]
        target = []

        for label in coarse_target_labels:
            count = 0

            for _, row in file_df.iterrows():
                if int(row['annotator_id']) == 0:
                    # If we have a validated annotation, just use that
                    count = row[label + '_presence']
                    break
                else:
                    count += row[label + '_presence']

            if count > 0:
                target.append(1.0)
            else:
                target.append(0.0)

        target_list[key] = target

    return target_list

def data_generator(data_dir, target_list, batch_size=64, random_state=20180123, start_batch_idx=None):

    random.seed(random_state)
    batch = None
    curr_batch_size = 0
    batch_idx = 0
        
    for fname in cycle_shuffle(os.listdir(data_dir)):
        data_path = os.path.join(data_dir, fname)

        blob_start_idx = 0
        data_blob = np.load(data_path)
        mel_blob = data_blob['db_mels']
        target_blob = np.array([target_list[fname] for _ in range(mel_blob.shape[0])])

        blob_size = mel_blob.shape[0]

        while blob_start_idx < blob_size:
            blob_end_idx = min(blob_start_idx + batch_size - curr_batch_size, blob_size)

            if start_batch_idx is None or batch_idx >= start_batch_idx:
                if batch is None:
                    batch = {'mel': mel_blob[blob_start_idx:blob_end_idx],\
                             'target': target_blob[blob_start_idx:blob_end_idx]}
                else:
                    batch['mel'] = np.concatenate([batch['mel'], mel_blob[blob_start_idx:blob_end_idx]])
                    batch['target'] = np.concatenate([batch['target'], target_blob[blob_start_idx:blob_end_idx]])

            curr_batch_size += blob_end_idx - blob_start_idx
            blob_start_idx = blob_end_idx

            if blob_end_idx == blob_size:
                data_blob.close()

            if curr_batch_size == batch_size:
                # If we are starting from a particular batch, skip yielding all
                # of the prior batches
                if start_batch_idx is None or batch_idx >= start_batch_idx:
                    batch['mel'] = np.array(batch['mel'])[:, :, :, np.newaxis]
                    #print(np.shape(np.array(batch['target']))) (64, 8)
                    #print(np.shape(batch['mel'])) #(64, 64, 51, 1)
                    yield batch

                batch_idx += 1
                curr_batch_size = 0
                batch = None
            
def single_epoch_data_generator(data_dir, target_list, epoch_size, **kwargs):
    while True:
        data_gen = data_generator(data_dir, target_list, **kwargs)
        for idx, item in enumerate(data_gen):
            yield item
            if (idx + 1) == epoch_size:
                break
                
def initialize_uninitialized_variables(sess):
    if hasattr(tf, 'global_variables'):
        variables = tf.compat.v1.global_variables() #tf.global_variables()
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
            sess.run(tf.compat.v1.variables_initializer(uninitialized_variables))
        else:
            sess.run(tf.compat.v1.initialize_variables(uninitialized_variables)) 
            
def restore_save_quantized_model(model_path, output_dir):
    
    K.clear_session()
    output_path = os.path.join(output_dir, 'frozen_pipeline_cmsis_mels_quant.pb')
    eval_graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True    
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
        exit(0)
        
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                                                                        eval_sess,
                                                                        eval_graph_def,
                                                                        [eval_model.output.op.name]
                                                                        )
        with open(output_path, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

            
def train_quantized_model(model_path, train_dir, valid_dir, output_dir, target_list, steps_per_epoch, \
                          valid_steps_per_epoch, loss=None, num_epochs=100, patience=500,\
                          learning_rate=1e-4, optimizer='adam'):
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = []
    train_gen = data_generator(train_dir, target_list)
    valid_gen = single_epoch_data_generator(valid_dir, target_list, valid_steps_per_epoch)

    train_gen = pescador.maps.keras_tuples(train_gen, 'mel', 'target')
    valid_gen = pescador.maps.keras_tuples(valid_gen, 'mel', 'target')
    
    K.clear_session()
    #train graph
    train_graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    train_sess = tf.Session(config=config, graph=train_graph)
    K.set_session(train_sess)
    
    with train_graph.as_default():
        # Set up callbacks
        #K.set_learning_phase(1)
        cb = []
        # checkpoint
        model_weight_file = os.path.join(output_dir, 'cmsis_quantized_model_best.h5')

        cb.append(keras.callbacks.ModelCheckpoint(model_weight_file,
                                                  save_weights_only=False,
                                                  save_best_only=True,
                                                  monitor='val_loss'))
        # early stopping
        cb.append(keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience))

        # monitor losses
        history_csv_file = os.path.join(output_dir, 'history.csv')
        cb.append(keras.callbacks.CSVLogger(history_csv_file, append=True,
                                            separator=','))

        loss = 'binary_crossentropy'
        opt = tf.train.AdamOptimizer(learning_rate) #keras.optimizers.Adam(lr=learning_rate)
    
        model = keras.models.load_model(model_path)
        #print(model.summary())
            
        model.compile(opt, loss=loss)
        tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=2)
        initialize_uninitialized_variables(train_sess)

        history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=num_epochs,\
                                      validation_steps=valid_steps_per_epoch,\
                                      validation_data=valid_gen, callbacks=cb, verbose=2)
    
        #save graph and checkpoints
        saver = tf.train.Saver()
        saver.save(train_sess, save_path=os.path.join(output_dir, os.path.basename(output_dir)))
    
        # Save history
        with open(os.path.join(output_dir, 'history.pkl'), 'wb') as fd:
            pickle.dump(history.history, fd)
            
    return history, output_dir

if __name__=='__main__':
    
    batch_size = 64
    epochs = 1500
    patience=100
    
    DATA_DIR = '/beegfs/dr2915/sonyc_ust'
    MODEL_DIR = '/scratch/sk7898/quantization/pipeline_cmsis/'
    annotation_path = os.path.join(DATA_DIR, 'annotations.csv')
    taxonomy_path = os.path.join(DATA_DIR, 'dcase-ust-taxonomy.yaml')
    
    model_path = os.path.join(MODEL_DIR, 'pipeline_cmsis_mels.h5')
    train_data_dir = os.path.join(DATA_DIR, 'db_mels/train')
    validation_data_dir = os.path.join(DATA_DIR, 'db_mels/validate')
    output_dir = MODEL_DIR
    
    target_list = get_file_targets(annotation_path, taxonomy_path)
    #print(target_list)
    
    steps_per_epoch = int(np.ceil(len(os.listdir(train_data_dir)) / batch_size))
    valid_steps_per_epoch = int(np.ceil(len(os.listdir(validation_data_dir)) / batch_size))
    #print(steps_per_epoch)
    #print(valid_steps_per_epoch)
    
    history, output_dir = train_quantized_model(model_path, train_data_dir, validation_data_dir, output_dir,\
                                                target_list, steps_per_epoch, valid_steps_per_epoch, \
                                                patience=patience, num_epochs=epochs)
    
    quantize_trained_model = os.path.join(output_dir, 'cmsis_quantized_model_best.h5')
    restore_save_quantized_model(quantize_trained_model, output_dir)
