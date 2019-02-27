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
from data.usc.us8k import generate_us8k_fold_data

import time

# Load embedding
weights_path = 'cnn_l3_melspec2_recent/model_best_valid_accuracy.h5'
model_type = 'cnn_L3_melspec2'
embedding_type = 'audio'
pooling_type = 'short'

l3embedding_model = load_embedding(weights_path, model_type, embedding_type, pooling_type)

# Featurization
fold_num = 5
features = 'l3'
metadata_path = 'UrbanSound8K/metadata/UrbanSound8K.csv'
data_dir = 'UrbanSound8K/audio'
dataset_output_dir = 'embeddings'
random_state = ''
hop_size = 0.1

for i in range(1000):
    generate_us8k_fold_data(metadata_path, data_dir, fold_num-1, dataset_output_dir,
                                    l3embedding_model=l3embedding_model,
                                    features=features, hop_size=hop_size)

