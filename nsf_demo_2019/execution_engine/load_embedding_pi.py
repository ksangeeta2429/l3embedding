from keras.models import Model
from keras.layers import Layer, InputSpec, Input, Conv2D, BatchNormalization, MaxPooling2D, MaxPooling1D, Flatten, Activation, Lambda, Reshape, concatenate, Dense
from kapre.time_frequency import Spectrogram, Melspectrogram
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import dtypes

def construct_cnn_L3_melspec2():
    """
    Constructs a model that replicates that used in Look, Listen and Learn

    Relja Arandjelovic and (2017). Look, Listen and Learn. CoRR, abs/1705.08168, .

    Returns
    -------
    model:  L3 CNN model
            (Type: keras.models.Model)
    inputs: Model inputs
            (Type: list[keras.layers.Input])
    outputs: Model outputs
            (Type: keras.layers.Layer)
    """
    vision_model, x_i, y_i = construct_cnn_L3_orig_inputbn_vision_model()
    audio_model, x_a, y_a = construct_cnn_L3_melspec2_audio_model()

    m = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'cnn_L3_kapredbinputbn')
    return m

def L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, model_name, layer_size=128):
    """
    Merges the audio and vision subnetworks and adds additional fully connected
    layers in the fashion of the model used in Look, Listen and Learn

    Relja Arandjelovic and (2017). Look, Listen and Learn. CoRR, abs/1705.08168, .

    Returns
    -------
    model:  L3 CNN model
            (Type: keras.models.Model)
    inputs: Model inputs
            (Type: list[keras.layers.Input])
    outputs: Model outputs
            (Type: keras.layers.Layer)
    """
    # Merge the subnetworks
    weight_decay = 1e-5
    y = concatenate([vision_model(x_i), audio_model(x_a)])
    y = Dense(layer_size, activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = Dense(2, activation='softmax',
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(weight_decay))(y)
    m = Model(inputs=[x_i, x_a], outputs=y)
    m.name = model_name

    return m, [x_i, x_a], y

def construct_cnn_L3_orig_inputbn_vision_model():
    """
    Constructs a model that replicates the vision subnetwork  used in Look,
    Listen and Learn

    Relja Arandjelovic and (2017). Look, Listen and Learn. CoRR, abs/1705.08168, .

    Returns
    -------
    model:  L3 CNN model
            (Type: keras.models.Model)
    inputs: Model inputs
            (Type: list[keras.layers.Input])
    outputs: Model outputs
            (Type: keras.layers.Layer)
    """
    weight_decay = 1e-5
    ####
    # Image subnetwork
    ####
    # INPUT
    x_i = Input(shape=(224, 224, 3), dtype='float32')
    y_i = BatchNormalization()(x_i)

    # CONV BLOCK 1
    n_filter_i_1 = 64
    filt_size_i_1 = (3, 3)
    pool_size_i_1 = (2, 2)
    y_i = Conv2D(n_filter_i_1, filt_size_i_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = Conv2D(n_filter_i_1, filt_size_i_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = Activation('relu')(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_1, strides=2, padding='same')(y_i)

    # CONV BLOCK 2
    n_filter_i_2 = 128
    filt_size_i_2 = (3, 3)
    pool_size_i_2 = (2, 2)
    y_i = Conv2D(n_filter_i_2, filt_size_i_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = Conv2D(n_filter_i_2, filt_size_i_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_2, strides=2, padding='same')(y_i)

    # CONV BLOCK 3
    n_filter_i_3 = 256
    filt_size_i_3 = (3, 3)
    pool_size_i_3 = (2, 2)
    y_i = Conv2D(n_filter_i_3, filt_size_i_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = Conv2D(n_filter_i_3, filt_size_i_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_3, strides=2, padding='same')(y_i)

    # CONV BLOCK 4
    n_filter_i_4 = 512
    filt_size_i_4 = (3, 3)
    pool_size_i_4 = (28, 28)
    y_i = Conv2D(n_filter_i_4, filt_size_i_4, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = Conv2D(n_filter_i_4, filt_size_i_4,
                 name='vision_embedding_layer', padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_4, padding='same')(y_i)
    y_i = Flatten()(y_i)

    m = Model(inputs=x_i, outputs=y_i)
    m.name = 'vision_model'

    return m, x_i, y_i

def construct_cnn_L3_melspec2_audio_model():
    """
    Constructs a model that replicates the audio subnetwork  used in Look,
    Listen and Learn

    Relja Arandjelovic and (2017). Look, Listen and Learn. CoRR, abs/1705.08168, .

    Returns
    -------
    model:  L3 CNN model
            (Type: keras.models.Model)
    inputs: Model inputs
            (Type: list[keras.layers.Input])
    outputs: Model outputs
            (Type: keras.layers.Layer)
    """
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
    n_filter_a_1 = 64
    filt_size_a_1 = (3, 3)
    pool_size_a_1 = (2, 2)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_1, strides=2)(y_a)

    # CONV BLOCK 2
    n_filter_a_2 = 128
    filt_size_a_2 = (3, 3)
    pool_size_a_2 = (2, 2)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_2, strides=2)(y_a)

    # CONV BLOCK 3
    n_filter_a_3 = 256
    filt_size_a_3 = (3, 3)
    pool_size_a_3 = (2, 2)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_3, strides=2)(y_a)

    # CONV BLOCK 4
    n_filter_a_4 = 512
    filt_size_a_4 = (3, 3)
    pool_size_a_4 = (32, 24)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4,
                 kernel_initializer='he_normal',
                 name='audio_embedding_layer', padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_4)(y_a)

    y_a = Flatten()(y_a)

    m = Model(inputs=x_a, outputs=y_a)
    m.name = 'audio_model'

    return m, x_a, y_a


def convert_audio_model_to_embedding(audio_model, x_a, model_type, pooling_type='original', unpruned_kd_model=False):
    """
    Given and audio subnetwork, return a model that produces the learned
    embedding

    Args:
        audio_model: audio subnetwork
        x_a: audio data input Tensor
        model_type: the model type string

    Returns:
        m: Model object
        x_a : audio data input Tensor
        y_a: embedding output Tensor
    """

    pooling = {
        'cnn_L3_orig': {
            'original': (8, 8),
            'short': (32, 24),
        },
        'cnn_L3_kapredbinputbn': {
            'original': (8, 8),
            'short': (32, 24),
        },
        'cnn_L3_melspec1': {
            'original': (4, 8),
            'short': (16, 24),
        },
        'cnn_L3_melspec2': {
            'original': (8, 8),
            'short': (32, 24),
            'kd_256': 2,
            'kd_128': 4,
        },
        'cnn_L3_melspec2_audioonly': {
            'original': (8, 8),
            'short': (32, 24),
            'kd_256': 2,
            'kd_128': 4,
        },
        'cnn_L3_melspec2_reduced_audioonly': {
            'original': (8, 8),
            'short': (32, 24),
            'kd_256': 2,
            'kd_128': 4,
        },
        'cnn_L3_melspec2_masked': {
            'original': (8, 8),
            'short': (32, 24),
        },
        'cnn_L3_melspec2_reduced': {
            'original': (8, 8),
            'short': (32, 24),
        }
    }

    if unpruned_kd_model:
        pool_size = pooling[model_type]['short']
        embedding_pool = pooling[model_type][pooling_type]
    else:
        pool_size = pooling[model_type][pooling_type]

    embed_layer = audio_model.get_layer('audio_embedding_layer')
    y_a = MaxPooling2D(pool_size=pool_size, padding='same')(embed_layer.output)

    if unpruned_kd_model:
        y_a = Reshape((y_a.shape[3], 1))(y_a)
        y_a = MaxPooling1D(pool_size=embedding_pool, border_mode='valid')(y_a)

    y_a = Flatten()(y_a)
    m = Model(inputs=x_a, outputs=y_a)

    return m, x_a, y_a


def load_model_pi(weights_path, model_type, return_io=False):
    """
    Loads an audio-visual correspondence model

    Args:
        weights_path:  Path to Keras weights file
                       (Type: str)
        model_type:    Name of model type
                       (Type: str)

    Keyword Args:
        src_num_gpus:   Number of GPUs the saved model uses
                        (Type: int)

        tgt_num_gpus:   Number of GPUs the loaded model will use
                        (Type: int)

        return_io:  If True, return input and output tensors
                    (Type: bool)

    Returns:
        model:  Loaded model object
                (Type: keras.engine.training.Model)
        x_i:    Input Tensor. Not returned if return_io is False.
                (Type: keras.layers.Input)
        y_i:    Embedding output Tensor/Layer. Not returned if return_io is False.
                (Type: keras.layers.Layer)
    """

    # Assert

    if model_type not in MODELS:
        raise ValueError('Invalid model type: "{}"'.format(model_type))

    m, inputs, output = MODELS[model_type]()

    #if src_num_gpus > 1:
    #    m = multi_gpu_model(m, gpus=src_num_gpus)
    m.load_weights(weights_path)

    #if tgt_num_gpus is not None and src_num_gpus != tgt_num_gpus:
    #    m, inputs, output = convert_num_gpus(m, inputs, output, model_type,
    #                                         src_num_gpus, tgt_num_gpus)

    if return_io:
        return m, inputs, output
    else:
        return m


def load_embedding(weights_path, model_type, embedding_type, pooling_type, kd_model=False, thresholds=None, include_layers=None, num_filters=None, return_io=False,
                   from_convlayer=8):
    """
    Loads an embedding model

    Args:
        weights_path:    Path to Keras weights file
                         (Type: str)
        model_type:      Name of model type
                         (Type: str)
        embedding_type:  Type of embedding to load ('audio' or 'vision')
                         (Type: str)
        pooling_type:    Type of pooling applied to final convolutional layer
                         (Type: str)
        from_convlayer:  Get embedding from convlayer# (default is 8)

    Keyword Args:
        src_num_gpus:   Number of GPUs the saved model uses
                        (Type: int)

        tgt_num_gpus:   Number of GPUs the loaded model will use
                        (Type: int)

        return_io:  If True, return input and output tensors
                    (Type: bool)

    Returns:
        model:  Embedding model object
                (Type: keras.engine.training.Model)
        x_i:    Input Tensor. Not returned if return_io is False.
                (Type: keras.layers.Input)
        y_i:    Embedding output Tensor/Layer. Not returned if return_io is False.
                (Type: keras.layers.Layer)
    """

    def relabel_embedding_layer(audio_model, embedding_layer_num):
        count = 1

        for layer in audio_model.layers:
            layer_name = layer.name

            if (layer_name[0:6] == 'conv2d' or layer_name == 'audio_embedding_layer'):
                # Rename the conv layers as conv_1, conv_2 .... conv_8, and relabel audio embedding layer
                if count == embedding_layer_num:
                    layer.name = 'audio_embedding_layer'
                else:
                    layer.name = 'conv_' + str(count)

                count += 1
        return audio_model

    if 'masked' in model_type:
        # Convert thresholds list to dictionary
        conv_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8']
        # TODO: Enable
        #thresholds = conv_keyval_lists_to_dict(conv_layers, thresholds)

        #m, inputs, output = load_new_model(weights_path, model_type, src_num_gpus=src_num_gpus,
        #                                  tgt_num_gpus=tgt_num_gpus, thresholds=thresholds, return_io=True)
    elif 'reduced' in model_type:
        print('Coming soon')
        # TODO: Enable
        #f = h5py.File(weights_path, 'r')
        #m, inputs, output = load_new_model(weights_path, model_type, src_num_gpus=src_num_gpus,
        #                                   tgt_num_gpus=tgt_num_gpus, include_layers=include_layers, num_filters=num_filters, return_io=True)
    else:
        m, inputs, output = load_model_pi(weights_path, model_type, return_io=True)

    if 'audio' in model_type:
        x_a = inputs
    else:
        x_i, x_a = inputs


    # Unused
    #if embedding_type == 'vision':
    #    m_embed_model = m.get_layer('vision_model')
    #    m_embed, x_embed, y_embed = VISION_EMBEDDING_MODELS[model_type](m_embed_model, x_i)

    if embedding_type == 'audio':
        if not 'audio' in model_type:
            m_embed_model = m.get_layer('audio_model')
        else:
            m_embed_model = m

        # m_embed, x_embed, y_embed = AUDIO_EMBEDDING_MODELS[model_type](m_embed_model, x_a)
        if from_convlayer==8:
            m_embed, x_embed, y_embed = convert_audio_model_to_embedding(m_embed_model, x_a, model_type, pooling_type, kd_model)
        else:
            m_embed, x_embed, y_embed = convert_audio_model_to_embedding(relabel_embedding_layer(m_embed_model, from_convlayer),
                                                                         x_a, model_type, pooling_type)
    else:
        raise ValueError('Invalid embedding type: "{}"'.format(embedding_type))

    if return_io:
        return m_embed, x_embed, y_embed
    else:
        return m_embed


MODELS = {
    #'cnn_L3_orig': construct_cnn_L3_orig,
    #'tiny_L3': construct_tiny_L3,
    #'cnn_L3_kapredbinputbn': construct_cnn_L3_kapredbinputbn,
    #'cnn_L3_melspec1': construct_cnn_L3_melspec1,
    'cnn_L3_melspec2': construct_cnn_L3_melspec2,
    #'cnn_L3_melspec2_audioonly': construct_cnn_L3_melspec2_audio_model
}