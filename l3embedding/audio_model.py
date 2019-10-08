from keras.models import Model
from keras.layers import Layer, InputSpec, Input, Conv2D, BatchNormalization, MaxPooling2D, MaxPooling1D, Flatten, Activation, Lambda, Reshape
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


class MaskedConv2D(Layer):
  def __init__(self,
               threshold,
               rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):

    super(MaskedConv2D, self).__init__(trainable=trainable,
                                       name=name,
                                       **kwargs)

    self.rank = rank
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank,
                                                    'dilation_rate')
    
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = InputSpec(ndim=self.rank + 2)
    self.threshold = threshold


  def build(self, input_shape):
    #input_shape = tensor_shape.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    self.kernel = self.add_weight(name='kernel',
                                  shape=kernel_shape,
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint,
                                  trainable=True,
                                  dtype=dtypes.float32)

                                      
    self.mask = K.cast(K.greater(K.abs(self.kernel), self.threshold), dtypes.float32)
    self.masked_kernel = math_ops.multiply(self.mask, self.kernel)

    if self.use_bias:
      self.bias = self.add_weight(name='bias',
                                  shape=(self.filters,),
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  trainable=True,
                                  dtype=dtypes.float32)
    else:
      self.bias = None

    self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
   
    '''
    self._convolution_op = nn_ops.Convolution(input_shape,
                                              filter_shape=self.kernel.get_shape(),
                                              dilation_rate=self.dilation_rate,
                                              strides=self.strides,
                                              padding=op_padding.upper(),
                                              data_format=conv_utils.convert_data_format(self.data_format, self.rank + 2))
    '''
    self.built = True

  def call(self, inputs):
    #outputs = self._convolution_op(inputs, self.masked_kernel)
    if self.rank == 1:
      outputs = K.conv1d(
                inputs,
                self.masked_kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
    if self.rank == 2:
      outputs = K.conv2d(
                inputs,
                self.masked_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
    if self.rank == 3:
      outputs = K.conv3d(
                inputs,
                self.masked_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(outputs)
    
    return outputs


  def compute_output_shape(self, input_shape):
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(space[i],
                                                self.kernel_size[i],
                                                padding=self.padding,
                                                stride=self.strides[i],
                                                dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return (input_shape[0],) + tuple(new_space) + (self.filters,)

  def get_config(self):
    config = {
      'threshold': self.threshold,
      'rank': self.rank,
      'filters': self.filters,
      'kernel_size': self.kernel_size,
      'strides': self.strides,
      'padding': self.padding,
      'data_format': self.data_format,
      'dilation_rate': self.dilation_rate,
      'activation': activations.serialize(self.activation),
      'use_bias': self.use_bias,
      'kernel_initializer': initializers.serialize(self.kernel_initializer),
      'bias_initializer': initializers.serialize(self.bias_initializer),
      'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer': regularizers.serialize(self.bias_regularizer),
      'activity_regularizer': regularizers.serialize(self.activity_regularizer),
      'kernel_constraint': constraints.serialize(self.kernel_constraint),
      'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(MaskedConv2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

def construct_cnn_L3_orig_audio_model():
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
    n_dft = 512
    #n_win = 480
    #n_hop = n_win//2
    n_hop = 242
    asr = 48000
    audio_window_dur = 1
    # INPUT
    x_a = Input(shape=(1, asr * audio_window_dur), dtype='float32')

    # SPECTROGRAM PREPROCESSING
    # 257 x 199 x 1
    y_a = Spectrogram(n_dft=n_dft, n_hop=n_hop, power_spectrogram=1.0, # n_win=n_win,
                      return_decibel_spectrogram=False, padding='valid')(x_a)

    # Apply normalization from L3 paper
    y_a = Lambda(lambda x: tf.log(tf.maximum(x, 1e-12)) / 5.0)(y_a)

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


def construct_cnn_L3_kapredbinputbn_audio_model():
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
    n_dft = 512
    #n_win = 480
    #n_hop = n_win//2
    n_hop = 242
    asr = 48000
    audio_window_dur = 1
    # INPUT
    x_a = Input(shape=(1, asr * audio_window_dur), dtype='float32')

    # SPECTROGRAM PREPROCESSING
    # 257 x 199 x 1
    y_a = Spectrogram(n_dft=n_dft, n_hop=n_hop, power_spectrogram=1.0, # n_win=n_win,
                      return_decibel_spectrogram=True, padding='valid')(x_a)
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

def construct_cnn_L3_melspec1_audio_model():
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
    n_mels = 128
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
    #y_a = BatchNormalization()(y_a)

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
    pool_size_a_4 = (16, 24)
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


def construct_cnn_L3_melspec2_masked_audio_model(thresholds):
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
    y_a = BatchNormalization(name='batch_normalization_1')(y_a)

    # CONV BLOCK 1
    n_filter_a_1 = 64
    filt_size_a_1 = (3, 3)
    pool_size_a_1 = (2, 2)
    y_a = MaskedConv2D(thresholds['conv_1'], 2, n_filter_a_1, filt_size_a_1, padding='same',
                       kernel_initializer='he_normal',
                       name='masked_conv2d_1',
                       kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization(name='batch_normalization_2')(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaskedConv2D(thresholds['conv_2'], 2, n_filter_a_1, filt_size_a_1, padding='same',
                       kernel_initializer='he_normal',
                       name='masked_conv2d_2',
                       kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization(name='batch_normalization_3')(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_1, strides=2)(y_a)

    # CONV BLOCK 2
    n_filter_a_2 = 128
    filt_size_a_2 = (3, 3)
    pool_size_a_2 = (2, 2)
    y_a = MaskedConv2D(thresholds['conv_3'],2, n_filter_a_2, filt_size_a_2, padding='same',
                       kernel_initializer='he_normal',
                       name='masked_conv2d_3',
                       kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization(name='batch_normalization_4')(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaskedConv2D(thresholds['conv_4'], 2, n_filter_a_2, filt_size_a_2, padding='same',
                       kernel_initializer='he_normal',
                       name='masked_conv2d_4',
                       kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization(name='batch_normalization_5')(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_2, strides=2)(y_a)

    # CONV BLOCK 3
    n_filter_a_3 = 256
    filt_size_a_3 = (3, 3)
    pool_size_a_3 = (2, 2)
    y_a = MaskedConv2D(thresholds['conv_5'], 2, n_filter_a_3, filt_size_a_3, padding='same',
                       kernel_initializer='he_normal',
                       name='masked_conv2d_5',
                       kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization(name='batch_normalization_6')(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaskedConv2D(thresholds['conv_6'], 2, n_filter_a_3, filt_size_a_3, padding='same',
                       kernel_initializer='he_normal',
                       name='masked_conv2d_6',
                       kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization(name='batch_normalization_7')(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_3, strides=2)(y_a)

    # CONV BLOCK 4
    n_filter_a_4 = 512
    filt_size_a_4 = (3, 3)
    pool_size_a_4 = (32, 24)
    y_a = MaskedConv2D(thresholds['conv_7'], 2, n_filter_a_4, filt_size_a_4, padding='same',
                       kernel_initializer='he_normal',
                       name='masked_conv2d_7',
                       kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization(name='batch_normalization_8')(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaskedConv2D(thresholds['conv_8'], 2, n_filter_a_4, filt_size_a_4,
                       kernel_initializer='he_normal',
                       name='audio_embedding_layer', padding='same',
                       kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization(name='batch_normalization_9')(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_4)(y_a)

    y_a = Flatten()(y_a)

    m = Model(inputs=x_a, outputs=y_a)
    m.name = 'audio_model'

    return m, x_a, y_a

def construct_cnn_L3_melspec2_kd_audio_model(masks):
    
    def get_masked_output(y, masks, layer_name):
        out = nn.convolution(input=tf.ones_like(y), filter=masks[layer_name], padding='SAME', data_format='NHWC')
        out = K.cast(K.greater(K.abs(out), 0.0), dtypes.float32)
        return out

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

    #y_a_1 = get_masked_output(y_a, masks, 'conv_1')

    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 name='conv_1',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)

    #y_a = Lambda(lambda x: math_ops.multiply(y_a_1, x), trainable=False)(y_a)

    y_a_2 = get_masked_output(y_a, masks, 'conv_2')

    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 name='conv_2',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)

    
    y_a = Lambda(lambda x: math_ops.multiply(y_a_2, x), trainable=False)(y_a)

    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_1, strides=2)(y_a)

    # CONV BLOCK 2
    n_filter_a_2 = 128
    filt_size_a_2 = (3, 3)
    pool_size_a_2 = (2, 2)
    y_a_3 = get_masked_output(y_a, masks, 'conv_3')

    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 name='conv_3',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)

    y_a = Lambda(lambda x: math_ops.multiply(y_a_3, x), trainable=False)(y_a)

    y_a_4 = get_masked_output(y_a, masks, 'conv_4')

    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 name='conv_4',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)

    y_a = Lambda(lambda x: math_ops.multiply(y_a_4, x), trainable=False)(y_a)

    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_2, strides=2)(y_a)

    # CONV BLOCK 3
    n_filter_a_3 = 256
    filt_size_a_3 = (3, 3)
    pool_size_a_3 = (2, 2)

    y_a_5 = get_masked_output(y_a, masks, 'conv_5')

    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 name='conv_5',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)

    y_a = Lambda(lambda x: math_ops.multiply(y_a_5, x), trainable=False)(y_a)

    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)

    y_a_6 = get_masked_output(y_a, masks, 'conv_6')

    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 name='conv_6',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)

    y_a = Lambda(lambda x: math_ops.multiply(y_a_6, x), trainable=False)(y_a)

    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_3, strides=2)(y_a)

    # CONV BLOCK 4
    n_filter_a_4 = 512
    filt_size_a_4 = (3, 3)
    pool_size_a_4 = (32, 24)

    y_a_7 = get_masked_output(y_a, masks, 'conv_7')

    y_a = Conv2D(n_filter_a_4, filt_size_a_4, padding='same',
                 kernel_initializer='he_normal',
                 name='conv_7',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)

    y_a = Lambda(lambda x: math_ops.multiply(y_a_7, x), trainable=False)(y_a)
        
    y_a_8 = get_masked_output(y_a, masks, 'conv_8')

    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4,
                 kernel_initializer='he_normal',
                 name='audio_embedding_layer', padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    
    y_a = Lambda(lambda x: math_ops.multiply(y_a_8, x), trainable=False)(y_a)
    
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_4)(y_a)

    y_a = Flatten()(y_a)

    m = Model(inputs=x_a, outputs=y_a)
    m.name = 'audio_model'

    return m, x_a, y_a


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
                     name='audio_embedding_layer', padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    
        y_a = BatchNormalization()(y_a)
        y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_4)(y_a)

    y_a = Flatten()(y_a)

    m = Model(inputs=x_a, outputs=y_a)
    m.name = 'audio_model'

    return m, x_a, y_a


def construct_cnn_L3_melspec2_audio_model(n_mels=256, n_hop = 242, n_dft = 2048,
                                          asr = 48000, fmax=None, halved_convs=False, audio_window_dur = 1):
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
    #n_win = 480
    #n_hop = n_win//2
    #n_mels = 256
    #n_hop = 242
    #asr = 48000
    #audio_window_dur = 1
    # INPUT
    x_a = Input(shape=(1, asr * audio_window_dur), dtype='float32')

    # MELSPECTROGRAM PREPROCESSING
    y_a = Melspectrogram(n_dft=n_dft, n_hop=n_hop, n_mels=n_mels,
                      sr=asr, fmax=fmax, power_melgram=1.0, htk=True, # n_win=n_win,
                      return_decibel_melgram=True, padding='same')(x_a)
    y_a = BatchNormalization()(y_a)

    #print('fmax:', fmax)

    # CONV BLOCK 1
    n_filter_a_1 = 64

    if halved_convs:
        n_filter_a_1 //= 2

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

    if halved_convs:
        n_filter_a_2 //= 2

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

    if halved_convs:
        n_filter_a_3 //= 2

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

    if halved_convs:
        n_filter_a_4 //= 2

    filt_size_a_4 = (3, 3)
    #pool_size_a_4 = (32, 24)
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

    pool_size_a_4 = tuple(y_a.get_shape().as_list()[1:3]) #(32, 24)
    y_a = MaxPooling2D(pool_size=pool_size_a_4)(y_a)

    #print('Pool Size: ', pool_size_a_4)

    y_a = Flatten()(y_a)

    m = Model(inputs=x_a, outputs=y_a)
    m.name = 'audio_model'

    #m.summary()

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
            '16k_64_50': (8,6),
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


def construct_tiny_L3_audio_model():
    """
    Constructs a model that implements a small L3 audio subnetwork

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
    n_dft = 512
    n_win = 480
    n_hop = n_win//2
    asr = 48000
    audio_window_dur = 1
    # INPUT
    x_a = Input(shape=(1, asr * audio_window_dur), dtype='float32')

    # SPECTROGRAM PREPROCESSING
    y_a = Spectrogram(n_dft=n_dft, n_win=n_win, n_hop=n_hop,
                      return_decibel_spectrogram=True, padding='valid')(x_a)

    y_a = Conv2D(10, (5,5), padding='valid', strides=(1,1),
                 kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=(3,3), strides=3)(y_a)
    y_a = Conv2D(10, (5,5), padding='valid', strides=(1,1),
                 kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=(3,3), strides=3)(y_a)
    y_a = Conv2D(10, (5,5), padding='valid', strides=(1,1),
                 kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=(3,3), strides=3)(y_a)
    y_a = Flatten(name='embedding')(y_a)
    m = Model(inputs=x_a, outputs=y_a)
    m.name = 'audio_model'

    return m, x_a, y_a
