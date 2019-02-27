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
from skimage import img_as_float
from audio import pcm2float
import h5py
from keras.models import Model
from model import *

def cycle_shuffle(iterable, shuffle=True):
    lst = list(iterable)
    while True:
        yield from lst
        if shuffle:
            random.shuffle(lst)

            
def smooth_categorical_crossentropy(target, output, from_logits=False, label_smoothing=0.0):
    """Categorical crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        _epsilon = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output)

    return tf.losses.softmax_cross_entropy(
        target, output, label_smoothing=label_smoothing)


def overlapping_time_slice_stack(x, ksize, stride, padding='SAME'):
    from tensorflow import extract_image_patches as extract

    ksizes = [1, 1, ksize, 1]
    strides = [1, 1, stride, 1]
    rates = [1, 1, 1, 1]

    x_slices = K.reshape(x, [-1, 1, K.int_shape(x)[-1], 1])
    x_slices = extract(x_slices, ksizes, strides, rates, padding)
    x_slices = K.squeeze(x_slices, axis=1)
    return x_slices


def construct_student_audio_model():
    num_conv2d=2
    asr = 48000
    audio_window_dur = 1
    input_size = asr * audio_window_dur

    input_layer = Input(shape=(1, input_size), dtype='float32')
    x = Lambda(lambda x: overlapping_time_slice_stack(x, 40, 20))(input_layer)
    x = Conv1D(filters=64, kernel_size=256, strides=1, use_bias=False, kernel_regularizer=l2(1e-5))(x)
    x = Lambda(lambda x: np.abs(x))(x)
    x = AveragePooling1D(pool_size=64, strides=1, padding='same')(x)
    x = Reshape([K.int_shape(x)[-2], K.int_shape(x)[-1], 1])(x)

    for i in range(num_conv2d):
        x = Conv2D(filters=32, kernel_size=(2, 2), padding='valid', use_bias=True, kernel_regularizer=l2(1e-5))(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='valid')(x)

    x = Flatten()(x)
    #x = Dense(num_classes, activation='softmax', use_bias=False, kernel_regularizer=l2(1e-5))(x)
    model = Model(inputs=input_layer, outputs=x)
    model.name = 'audio_model'
    return model, input_layer, x


def load_model_with_student_audio():
    vision_model, x_i, y_i = construct_cnn_L3_orig_vision_model() #construct_cnn_L3_orig_inputbn_vision_model()
    audio_model, x_a, y_a = construct_student_audio_model()

    m, inputs, outputs = L3_merge_audio_vision_models(vision_model, x_i, audio_model, x_a, 'cnn_L3_cnn_melspec1')
    return m


def single_epoch_data_generator(data_dir, soft_labels_dir, epoch_size, **kwargs):
    while True:
        data_gen = data_generator(data_dir, soft_labels_dir, **kwargs)
        for idx, item in enumerate(data_gen):
            yield item
            # Once we generate all batches for an epoch, restart the generator
            if (idx + 1) == epoch_size:
                break


def get_labels(label_files, file_idx, last_processed_label_idx, labels):
    if len(labels) > 0:
        temp_labels = labels[last_processed_label_idx:len(labels)-1]
        labels[:] = []
        labels.append(temp_labels)

    with open(label_files[file_idx]) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                labels.append(row)

    file_idx += 1
    return labels, file_idx


def data_generator(data_dir, soft_labels_dir, batch_size=512, random_state=20180123, start_batch_idx=None, keys=None):
    random.seed(random_state)

    batch = None
    label_files = []
    labels = []
    curr_batch_size = 0
    batch_idx = 0
    file_idx = 0
    start_label_idx = 0

    # Limit keys to avoid producing batches with all of the metadata fields
    if not keys:
        keys = ['audio', 'video']

    for soft_label_file in os.listdir(soft_labels_dir):
        file_path = os.path.join(soft_labels_dir, soft_label_file)
        label_files.append(file_path)

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

                    if ((batch_idx+1)*batch_size) > len(labels):
                        start_label_idx = 0
                        # find the last index of processed label
                        last_processed_label_idx = (batch_idx * batch_size)-1
                        labels, file_idx = get_labels(label_files, file_idx, last_processed_label_idx, labels)
                        batch['label'] = labels[start_label_idx:(start_label_idx+batch_size)]
                    else:
                        batch['label'] = labels[start_label_idx:(start_label_idx+batch_size)]
                        start_label_idx += batch_size
                    yield batch

                batch_idx += 1
                curr_batch_size = 0
                batch = None


def train(train_data_dir, validation_data_dir, soft_labels_train_dir, soft_labels_val_dir, num_epochs=150, validation_epoch_size=1024,
          train_epoch_size=4096, train_batch_size=16, learning_rate=1e-4, validation_batch_size=64,
          random_state=20180123, gpus=1):

    train_gen = data_generator(
        train_data_dir,
        soft_labels_train_dir,
        batch_size=train_batch_size,
        random_state=random_state,
        start_batch_idx=None)

    train_gen = pescador.maps.keras_tuples(train_gen,
                                           ['video', 'audio'],
                                           'label')

    val_gen = single_epoch_data_generator(
        validation_data_dir,
        soft_labels_val_dir,
        validation_epoch_size,
        batch_size=validation_batch_size,
        random_state=random_state)

    val_gen = pescador.maps.keras_tuples(val_gen,
                                         ['video', 'audio'],
                                         'label')

    student = load_model_with_student_audio()
    optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)

    student.compile(optimizer=optimizer,
                    loss=lambda y_true, y_pred: smooth_categorical_crossentropy(y_true, y_pred, label_smoothing=0.1),
                    metrics=[keras.metrics.categorical_accuracy])

    history = student.fit_generator(train_gen, train_epoch_size, num_epochs,
                                    validation_data=val_gen,
                                    validation_steps=validation_epoch_size,
                                    verbose=1)

    print(history)
    return history

train_data_dir = '/beegfs/work/AudioSetSamples/music_train' # _environmental/urban_train'
validation_data_dir = '/beegfs/work/AudioSetSamples/music_valid' # _environmental/urban_valid'
soft_labels_train_dir = '/home/sk7898/l3embedding/l3embedding/train_labels'
soft_labels_val_dir = '/home/sk7898/l3embedding/l3embedding/val_labels'
history = train(train_data_dir, validation_data_dir, soft_labels_train_dir, soft_labels_val_dir)
