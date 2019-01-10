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
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

##########
# Pruning Version 1 :
# Step 1: Prune without fine-tuning [with different sparsity levels] and the threshold values are decided by sparsity
#         a) Same (Block): Prune each with same sparsity value
#         b) Variable (Block): Prune in three fashion: 1) less more more more 2) less more more less 3) monotonically increasing
#         c) Layer wise: with information from 1b
# Step 2: Fine-tune the whole model at a go
# Step 3: Fine-tune the model CONV block wise

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
# Pruning Version 2 : Prune whole feature-maps
# Pseudo-code
#
##########

def cycle_shuffle(iterable, shuffle=True):
    lst = list(iterable)
    while True:
        yield from lst
        if shuffle:
            random.shuffle(lst)


def data_generator(data_dir, batch_size=512, random_state=20180123,
                   start_batch_idx=None, keys=None):
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
                # If we are starting from a particular batch, skip yielding all
                # of the prior batches
                if start_batch_idx is None or batch_idx >= start_batch_idx:
                    # Preprocess video so samples are in [-1,1]
                    batch['video'] = 2 * img_as_float(batch['video']).astype('float32') - 1

                    # Convert audio to float
                    batch['audio'] = pcm2float(batch['audio'], dtype='float32')

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
        if ('conv_' in layer.name and sparsity_dict[layer.name]):
            target_weights = np.empty_like(layer.get_weights())
            weights = layer.get_weights()[0]
            threshold = calculate_threshold(weights, sparsity_dict[layer.name])
            mask      = K.cast(K.greater(K.abs(weights), threshold), dtypes.float32)

            masks[layer.name] = mask
            new_weights = weights * K.eval(mask)
            target_weights[0] = new_weights
            
            target_weights[1] = layer.get_weights()[1]
            layer.set_weights(target_weights)

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

    val_gen = single_epoch_data_generator(
        validation_data_dir,
        validation_epoch_size,
        batch_size=validation_batch_size,
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


def train(model_to_retrain, train_data_dir, validation_data_dir, finetune=False, output_dir = None, \
          num_epochs=300, train_epoch_size=4096, validation_epoch_size=1024, train_batch_size=64, validation_batch_size=64,\
          model_type = 'cnn_L3_melspec2', random_state=20180216, learning_rate=0.001, verbose=True, \
          checkpoint_interval=10, gpus=1, continue_model_dir=None):

    if output_dir is None:
        if finetune:
            output_dir = '/scratch/sk7898/pruning_finetune_output'
        else:
            output_dir = '/scratch/sk7898/pruning_kd_output'

    # Form model ID
    data_subset_name = os.path.basename(train_data_dir)
    data_subset_name = data_subset_name[:data_subset_name.rindex('_')]
    model_id = os.path.join(data_subset_name, model_type)

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

    # Make sure the directories we need exist
    if continue_model_dir:
        model_dir = continue_model_dir
    else:
        model_dir = os.path.join(output_dir, 'embedding', model_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        
    if continue_model_dir:
        latest_model_path = os.path.join(continue_model_dir, 'model_latest.h5')
        model = load_model(latest_model_path)
    else:
        model = model_to_retrain

    param_dict['model_dir'] = model_dir
    train_config_path = os.path.join(model_dir, 'config.json')
    with open(train_config_path, 'w') as fd:
        json.dump(param_dict, fd, indent=2)

    if finetune:
        model.compile(Adam(lr=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile(Adam(lr=learning_rate),
                      loss='mean_squared_error',
                      metrics=['mae'])

    # Save the model
    model_json_path = os.path.join(model_dir, 'model.json')
    model_json = model.to_json()
    with open(model_json_path, 'w') as fd:
        json.dump(model_json, fd, indent=2)

    latest_weight_path = os.path.join(model_dir, 'model_latest.h5')
    best_valid_loss_weight_path = os.path.join(model_dir, 'model_best_valid_loss.h5')
    checkpoint_weight_path = os.path.join(model_dir, 'model_checkpoint.{epoch:02d}.h5')

    # Load information about last epoch for initializing callbacks and data generators
    if continue_model_dir is not None:
        prev_train_hist_path = os.path.join(continue_model_dir, 'history_csvlog.csv')
        last_epoch_idx, last_val_loss = get_restart_info(prev_train_hist_path)

    # Set up callbacks
    cb = []
    cb.append(keras.callbacks.ModelCheckpoint(latest_weight_path,
                                              save_weights_only=False,
                                              verbose=1))


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

    # Save history
    with open(os.path.join(model_dir, 'history.pkl'), 'wb') as fd:
        pickle.dump(history.history, fd)

    return history


def initialize_weights(model, sparse_model, is_L3=True):
    if is_L3:
        audio_model = model.get_layer('audio_model')

        for layer in audio_model.layers:
            print(layer.name)
        


def retrain(l3_model, masks, train_data_dir, validation_data_dir, finetune=False):
    if finetune:
        model = construct_cnn_L3_melspec2_kd(masks)
        model = initialize_weights(model, l3_model, is_L3=True)  
        train(model, train_data_dir, validation_data_dir, finetune=finetune)
    else:
        audio_model = construct_cnn_L3_melspec2_kd_audio_model(masks) 
        audio_model = initialize_weights(audio_model, l3_model.get_layer('audio_model'), is_L3=False)
        train(audio_model, validation_data_dir, finetune=finetune)


def zero_check(sparsity_list):
    for val in sparsity_list:
        if(val):
            return False
    return True


def printList(sparsity_list):
    printStr = "Sparsity Values: "
    for val in sparsity_list:
        printStr += str(val) + ", "
    print(printStr)


def pruning(weight_path, train_data_dir, validation_data_dir, output_dir = '/scratch/sk7898/pruned_model', blockwise=False,\
            layerwise=False, per_layer=False, test_model=False, save_model=False, retrain_model=False):
    
    conv_blocks = 4
    
    sparsity_values = [70., 80., 85., 90., 95.]
    sparsity_blks = [0, 0, 0, 0]
    if per_layer:
        sparsity_layers = [0, 0, 0, 0, 0, 0, 0, 0]
    else:
        sparsity_layers = [[0, 30., 40., 50., 30., 50., 50., 60.]]
        ''',\
                           [0, 40., 50., 60., 40., 60., 60., 70.],\
                           [0, 40., 50., 60., 40., 70., 70., 80.]]

        
                           [0, 60., 60., 70., 50., 70., 70., 80.],\
                           [0, 70., 70., 75., 60., 80., 80., 85.],\
                           [0, 80., 80., 85., 40., 85., 85., 95.]]
        '''
    if(blockwise and zero_check(sparsity_blks)):
        print("Block-wise Pruning")
        print("*********************************************************************")
    
        for sparsity in sparsity_values:
            for blockid in range(conv_blocks):
                model, audio_model = load_audio_model_for_pruning(weight_path)
                sparsity_vals = get_sparsity_blocks(blockid, sparsity, sparsity_blks) 
                sparsified_model = sparsify_block(audio_model, sparsity_vals)
            
                model.get_layer('audio_model').set_weights(sparsified_model.get_weights()) 
            
                score = test(model, validation_dir)
                print('Conv Block Pruned: {0} Sparsity Value: {1}'.format(blockid+1, sparsity))
                print('Loss: {0} Accuracy: {1}'.format(score[0], score[1]))     
                print('---------------------------------------------------------------')
    
    if(layerwise and per_layer):
        print("Layer-wise Pruning")
        print("**********************************************************************")
        for sparsity in sparsity_values:
            for layerid in range(conv_blocks*2):
                model, audio_model = load_audio_model_for_pruning(weight_path)
                sparsity_vals = get_sparsity_layers(layerid, sparsity, sparsity_layers)
                sparsified_model, masks = sparsify_layer(audio_model, sparsity_vals)

                model.get_layer('audio_model').set_weights(sparsified_model.get_weights())

                score = test(model, validation_dir)
                print('Conv Layer Pruned: {0} Sparsity Value: {1}'.format(layerid+1, sparsity))
                print('Loss: {0} Accuracy: {1}'.format(score[0], score[1]))
                print('----------------------------------------------------------------')
    
    if(layerwise and not per_layer):
        print("Specific Pruning Values")
        print("**********************************************************************")
        for sparsity in sparsity_layers: 
            model, audio_model = load_audio_model_for_pruning(weight_path)
            sparsity_vals = get_sparsity_layers(None, None, sparsity)
            sparsified_model, masks = sparsify_layer(audio_model, sparsity_vals)
            
            model.get_layer('audio_model').set_weights(sparsified_model.get_weights())
            if test_model:
                score = test(model, validation_dir)
                printList(sparsity)
                print('Loss: {0} Accuracy: {1}'.format(score[0], score[1]))
                print('----------------------------------------------------------------')
            
            if save_model:
                pruned_model_name = 'pruned_audio_'+str(score[1])+'.h5'
                pruned_model_path = os.path.join(output_dir, pruned_model_name)
                sparsified_model.save(pruned_model_path)

            if retrain_model:
                retrain(model, masks, train_data_dir, validation_data_dir, finetune=True)


weight_path = '/home/sk7898/l3embedding/models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5'
train_data_dir = '/beegfs/work/AudioSetSamples/music_train'
validation_data_dir = '/beegfs/work/AudioSetSamples/music_valid'

pruning(weight_path, train_data_dir, validation_data_dir, blockwise=False, layerwise=True, per_layer=False, retrain_model=True)
