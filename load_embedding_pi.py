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
from l3embedding.model import load_embedding

import time

weights_path = 'cnn_l3_melspec2_recent/model_best_valid_accuracy.h5'
model_type = 'cnn_L3_melspec2'
embedding_type = 'audio'
pooling_type = 'short'

# Load embedding


print('Starting to profile...')
start = time.time()

# Featurization
embedding = load_embedding(weights_path, model_type, embedding_type, pooling_type)
done = time.time()
elapsed = done - start
print(elapsed)

