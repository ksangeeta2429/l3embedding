import os
import random
import csv
import numpy as np
from keras.regularizers import l2
import tensorflow as tf
import keras
from keras.optimizers import Adam
import pescador
from keras.layers.recurrent import GRU
from keras.layers import *
from audio import pcm2float
import h5py
from keras.models import Model
from model import *

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
# Construct the l3 model and load the weights from trained cnn_l3_melspec1
# Version 1
# Run a loop for different sparsity values [30, 40, 50, 60, 70] for each CONV block and test the performance
# Version 2
# Run a loop for different sparsity values [30, 40, 50, 60, 70] for each CONV layer and test the performance
##########

##########
# Pruning Version 2 : Prune whole feature-maps
# Pseudo-code
#
##########

##########
# Pruning Version 3 : Combination of pruning both feature-maps and weights of leftover feature map
# Pseudo-code
#
##########

# def construct_cnn_L3_melspec1_audio_model():
#     """
#     Constructs a model that replicates the audio subnetwork  used in Look,
#     Listen and Learn
#
#     Relja Arandjelovic and (2017). Look, Listen and Learn. CoRR, abs/1705.08168, .
#
#     Returns
#     -------
#     model:  L3 CNN model
#             (Type: keras.models.Model)
#     inputs: Model inputs
#             (Type: list[keras.layers.Input])
#     outputs: Model outputs
#             (Type: keras.layers.Layer)
#     """
#     weight_decay = 1e-5
#     ####
#     # Audio subnetwork
#     ####
#     n_dft = 2048
#     #n_win = 480
#     #n_hop = n_win//2
#     n_mels = 128
#     n_hop = 242
#     asr = 48000
#     audio_window_dur = 1
#     # INPUT
#     x_a = Input(shape=(1, asr * audio_window_dur), dtype='float32')
#
#     # MELSPECTROGRAM PREPROCESSING
#     # 128 x 199 x 1
#     y_a = Melspectrogram(n_dft=n_dft, n_hop=n_hop, n_mels=n_mels,
#                       sr=asr, power_melgram=1.0, htk=True, # n_win=n_win,
#                       return_decibel_melgram=True, padding='same')(x_a)
#     #y_a = BatchNormalization()(y_a)
#
#     # CONV BLOCK 1
#     n_filter_a_1 = 64
#     filt_size_a_1 = (3, 3)
#     pool_size_a_1 = (2, 2)
#     y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=regularizers.l2(weight_decay))(y_a)
#     y_a = BatchNormalization()(y_a)
#     y_a = Activation('relu')(y_a)
#     y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=regularizers.l2(weight_decay))(y_a)
#     y_a = BatchNormalization()(y_a)
#     y_a = Activation('relu')(y_a)
#     y_a = MaxPooling2D(pool_size=pool_size_a_1, strides=2)(y_a)
#
#     # CONV BLOCK 2
#     n_filter_a_2 = 128
#     filt_size_a_2 = (3, 3)
#     pool_size_a_2 = (2, 2)
#     y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=regularizers.l2(weight_decay))(y_a)
#     y_a = BatchNormalization()(y_a)
#     y_a = Activation('relu')(y_a)
#     y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=regularizers.l2(weight_decay))(y_a)
#     y_a = BatchNormalization()(y_a)
#     y_a = Activation('relu')(y_a)
#     y_a = MaxPooling2D(pool_size=pool_size_a_2, strides=2)(y_a)
#
#     # CONV BLOCK 3
#     n_filter_a_3 = 256
#     filt_size_a_3 = (3, 3)
#     pool_size_a_3 = (2, 2)
#     y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=regularizers.l2(weight_decay))(y_a)
#     y_a = BatchNormalization()(y_a)
#     y_a = Activation('relu')(y_a)
#     y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=regularizers.l2(weight_decay))(y_a)
#     y_a = BatchNormalization()(y_a)
#     y_a = Activation('relu')(y_a)
#     y_a = MaxPooling2D(pool_size=pool_size_a_3, strides=2)(y_a)
#
#     # CONV BLOCK 4
#     n_filter_a_4 = 512
#     filt_size_a_4 = (3, 3)
#     pool_size_a_4 = (16, 24)
#     y_a = Conv2D(n_filter_a_4, filt_size_a_4, padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=regularizers.l2(weight_decay))(y_a)
#     y_a = BatchNormalization()(y_a)
#     y_a = Activation('relu')(y_a)
#     y_a = Conv2D(n_filter_a_4, filt_size_a_4,
#                  kernel_initializer='he_normal',
#                  name='audio_embedding_layer', padding='same',
#                  kernel_regularizer=regularizers.l2(weight_decay))(y_a)
#     y_a = BatchNormalization()(y_a)
#     y_a = Activation('relu')(y_a)
#     y_a = MaxPooling2D(pool_size=pool_size_a_4)(y_a)
#
#     y_a = Flatten()(y_a)
#
#     m = Model(inputs=x_a, outputs=y_a)
#     m.name = 'audio_model'
#
#     return m, x_a, y_a
#
# def construct_pruned_audio_model():
#     for model_name, model_weights in models.items():
#         base_model_name = model_name
#         for sparsity in [50.,60.,70.,80.,90.]:
#             # load the pretrained model
#             model_name, model_weights = load_best(model_name, model_weights)
#             model_name = model_name + str(sparsity)
#
#             best_acc = 0.
#
#             # sparsify
#             model_weights = sparsify(model_weights, sparsity)
#
#             # train with 0.01
#             #best_acc = finetune(model_weights, best_acc, 30, 0.01)
#             # train with 0.001
#             #best_acc = finetune(model_weights, best_acc, 30, 0.001)
#
#         new_model = compress_convs(model_weights, compressed_models[base_model_name])
#
#         # finetune again - this is just to save the model
#         finetune(new_model, 0., 10, 0.001)
#
# def load_model_with_pruned_audio():
#     vision_model, x_i, y_i = construct_cnn_L3_orig_vision_model() #construct_cnn_L3_orig_inputbn_vision_model()
#     audio_model, x_a, y_a = construct_cnn_L3_melspec1_audio_model()
#
#     m, inputs, outputs = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'cnn_L3_cnn_melspec1')
#     return m
#
# def prune_block():

def pruning(model_type = 'cnn_L3_melspec1'):
    weight_path = '/home/sk7898/l3embedding/models/cnn_l3_melspec1/model_best_valid_accuracy_1gpu.h5'
    m, inputs, outputs = load_model(weight_path, model_type, return_io=True, src_num_gpus=1)

    for layer in model.layers:
        print (layer.name)

pruning()
