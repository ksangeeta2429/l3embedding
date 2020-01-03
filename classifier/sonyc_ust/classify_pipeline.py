import argparse
import csv
import datetime
import json
import gzip
import os
import sys
import pprint
import pickle as pk
import numpy as np
import pandas as pd
import oyaml as yaml

import keras
import tensorflow as tf
from keras.layers import Input, Dense, TimeDistributed, GlobalAveragePooling1D, BatchNormalization
from keras.models import Model, load_model
from keras import regularizers
from keras.optimizers import Adam
import keras.backend as K
from autopool import AutoPool1D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from balancing import mlsmote, lssmote
from scipy_optimizer import ScipyOpt


## HELPERS

def load_raw_audio(file_list, emb_dir):
    """
    Load saved embeddings from an embedding directory

    Parameters
    ----------
    file_list
    emb_dir

    Returns
    -------
    embeddings
    ignore_idxs

    """
    embeddings = []
    min_num_frames = None
    for idx, filename in enumerate(file_list):
        fname = os.path.splitext(filename)[0]
        emb_path = os.path.join(emb_dir, fname + '.npy.gz')
        if os.path.exists(emb_path):
            with gzip.open(emb_path, 'rb') as f:
                emb = np.load(f)
                embeddings.append(emb)
        else:
            emb_path = os.path.join(emb_dir, fname + '.npy')
            if os.path.exists(emb_path):
                with open(emb_path, 'rb') as f:
                    emb = np.load(f)
                    embeddings.append(emb)
            else:
                emb_path = os.path.join(emb_dir, fname + '.npz')
                if not os.path.exists(emb_path):
                    raise ValueError('Could not find embedding for {} in {}'.format(fname, emb_dir))

                with open(emb_path, 'rb') as f:
                    emb = np.load(f)['audio']
                    embeddings.append(emb)

        if min_num_frames is None or emb.shape[0] < min_num_frames:
            if min_num_frames is not None:
                err_msg = "Encountered mismatch of numbers of frames ({}, {})"
                print(err_msg.format(emb.shape[0], min_num_frames))
                sys.stdout.flush()
            min_num_frames = emb.shape[0]

    # Make sure all files have the same length
    embeddings = [emb[:min_num_frames, :] for emb in embeddings]

    return embeddings


def get_subset_split(annotation_data, split_path=None):
    """
    Get indices for train and validation subsets

    Parameters
    ----------
    annotation_data
    split_path

    Returns
    -------
    train_idxs
    valid_idxs

    """

    # Get the audio filenames and the splits without duplicates
    data = annotation_data[['split', 'sensor_id', 'audio_filename']].drop_duplicates().sort_values('audio_filename')

    custom_split = None
    if split_path:
        custom_split = {'train': [], 'validate': []}

        split_df = pd.read_csv(split_path)
        for _, row in split_df.iterrows():
            custom_split[row['split']].append(row['sensor_id'])

    train_idxs = []
    valid_idxs = []
    test_idxs = []
    for idx, (_, row) in enumerate(data.iterrows()):
        if not custom_split:
            if row['split'] == 'train':
                train_idxs.append(idx)
            elif row['split'] == 'validate':
                valid_idxs.append(idx)
            else:
                test_idxs.append(idx)
        else:
            if row['sensor_id'] in custom_split['train']:
                train_idxs.append(idx)
            elif row['sensor_id'] in custom_split['validate']:
                valid_idxs.append(idx)
            else:
                test_idxs.append(idx)

    return np.array(train_idxs), np.array(valid_idxs)


def get_file_targets(annotation_data, labels):
    """
    Get file target annotation vector for the given set of labels

    Parameters
    ----------
    annotation_data
    labels

    Returns
    -------
    target_list

    """
    target_list = []
    file_list = annotation_data.sort_values('audio_filename')['audio_filename'].unique().tolist()

    for filename in file_list:
        file_df = annotation_data[annotation_data['audio_filename'] == filename]
        target = []

        for label in labels:
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

        target_list.append(target)

    return np.array(target_list)


def get_file_sensor_targets(annotation_data):
    target_list = []

    sensor_ref = sorted(annotation_data.sort_values('sensor_id')['sensor_id'].unique().tolist())
    num_sensors = len(sensor_ref)

    rows = annotation_data.sort_values('audio_filename')[['audio_filename', 'sensor_id']].drop_duplicates().iterrows()

    for _, row in rows:
        filename = row['audio_filename']
        sensor_id = row['sensor_id']
        target_idx = sensor_ref.index(sensor_id)
        target = np.zeros((num_sensors,)).astype('float32')
        target[target_idx] = 1.0
        target_list.append(target)

    return np.array(target_list)


def get_file_proximity_targets(annotation_data, labels):
    target_list = []
    file_list = annotation_data.sort_values('audio_filename')['audio_filename'].unique().tolist()

    for filename in file_list:
        file_df = annotation_data[annotation_data['audio_filename'] == filename]
        target = []

        for label in labels:
            near = 0
            far = 0

            for _, row in file_df.iterrows():
                prox = row[label + '_proximity']
                if prox == 'near':
                    near += 1
                elif prox == 'far':
                    far += 1

            if near == 0 and far == 0 or near == far:
                target.append(0.0)
                target.append(0.0)
            elif near > far:
                target.append(1.0)
                target.append(0.0)
            elif far > near:
                target.append(0.0)
                target.append(1.0)

        target_list.append(target)

    return np.array(target_list)


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Courtesy of https://stackoverflow.com/a/42797620

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


## MODEL CONSTRUCTION


def construct_mlp_framewise(cnn, num_classes, sensor_factor=True,
                            num_sensors=None, proximity_factor=True,
                            num_proximity_classes=None,
                            hidden_layer_size=128, num_hidden_layers=0,
                            l2_reg=1e-5, activation='sigmoid'):
    """
    Construct a 2-hidden-layer MLP model for framewise processing

    Parameters
    ----------
    emb_size
    num_classes
    hidden_layer_size
    num_hidden_layers
    l2_reg

    Returns
    -------
    model

    """
    # First set all layers of cnn to non-trainable
    for l in cnn.layers:
        l.trainable=False

    # Input layer
    #inp = Input(shape=(emb_size,), dtype='float32', name='input')
    y = cnn.output

    # Add a batchnorm layer
    y = BatchNormalization(name='batch_normalization_dst')(y)

    # Add hidden layers
    for idx in range(num_hidden_layers):
        y = Dense(hidden_layer_size, activation='relu',
                  kernel_regularizer=regularizers.l2(l2_reg),
                  name='dense_{}'.format(idx+1))(y)

    if sensor_factor:
        assert num_sensors is not None

        y_sensor = Dense(num_sensors, activation='softmax',
                         name='sensor_output',
                         kernel_regularizer=regularizers.l2(l2_reg))(y)

    if proximity_factor:
        proximity_outputs = []
        for idx in range(num_proximity_classes):
            prox_output = Dense(2, activation='softmax',
                                name='proximity_{}'.format(idx),
                                kernel_regularizer=regularizers.l2(l2_reg))(y)
            proximity_outputs.append(prox_output)

        y_proximity = keras.layers.Concatenate(name='proximity_output')(proximity_outputs)

    # Concatenate
    if sensor_factor:
        y = keras.layers.Concatenate(name='concat_sensor')([y, y_sensor])
    if proximity_factor:
        y = keras.layers.Concatenate(name='concat_proximity')([y, y_proximity])

    # Output layer
    y = Dense(num_classes, activation=activation,
              kernel_regularizer=regularizers.l2(l2_reg), name='output')(y)

    if sensor_factor or proximity_factor:
        outputs = [y]
        if sensor_factor:
            outputs.append(y_sensor)
        if proximity_factor:
            outputs.append(y_proximity)
    else:
        outputs = y

    m = Model(inputs=cnn.input, outputs=outputs)
    m.name = 'urban_sound_classifier'
    print(m.summary())

    return m


def construct_mlp_mil(num_frames, emb_size, num_classes, sensor_factor=False,
                      num_sensors=None, proximity_factor=False, num_proximity_classes=None,
                      hidden_layer_size=128, num_hidden_layers=0, l2_reg=1e-5):
    """
    Construct a 2-hidden-layer MLP model for MIL processing

    Parameters
    ----------
    num_frames
    emb_size
    num_classes
    hidden_layer_size
    num_hidden_layers
    l2_reg

    Returns
    -------
    model

    """
    # Input layer
    inp = Input(shape=(num_frames, emb_size), dtype='float32', name='input')
    y = inp

    # Add hidden layers
    repr_size = emb_size
    for idx in range(num_hidden_layers):
        y = TimeDistributed(Dense(hidden_layer_size, activation='relu',
                                  kernel_regularizer=regularizers.l2(l2_reg)),
                            name='dense_{}'.format(idx+1),
                            input_shape=(num_frames, repr_size))(y)
        repr_size = hidden_layer_size

    if sensor_factor:
        assert num_sensors is not None

        y_sensor_t = TimeDistributed(Dense(num_sensors, activation='softmax',
                                           kernel_regularizer=regularizers.l2(l2_reg)),
                                     name='sensor_t',
                                     input_shape=(num_frames, repr_size))(y)
        y_sensor = GlobalAveragePooling1D(name='sensor_output')(y_sensor_t)

    if proximity_factor:
        proximity_outputs = []
        for idx in range(num_proximity_classes):
            prox_output = TimeDistributed(Dense(2, activation='softmax',
                                                kernel_regularizer=regularizers.l2(l2_reg)),
                                          name='proximity_{}'.format(idx),
                                          input_shape=(num_frames, repr_size))(y)
            proximity_outputs.append(prox_output)
        y_proximity_t = keras.layers.Concatenate(name='proximity_t')(proximity_outputs)
        y_proximity = GlobalAveragePooling1D(name='proximity_output')(y_proximity_t)

    # Concatenate
    if sensor_factor:
        y = keras.layers.Concatenate(name='concat_sensor')([y, y_sensor_t])
    if proximity_factor:
        y = keras.layers.Concatenate(name='concat_proximity')([y, y_proximity_t])

    # Output layer
    y = TimeDistributed(Dense(num_classes, activation='sigmoid',
                              kernel_regularizer=regularizers.l2(l2_reg)),
                        name='output_t',
                        input_shape=(num_frames, repr_size))(y)

    # Apply autopool over time dimension
    y = AutoPool1D(kernel_constraint=keras.constraints.non_neg(),
                   axis=1, name='output')(y)

    if sensor_factor or proximity_factor:
        outputs = [y]
        if sensor_factor:
            outputs.append(y_sensor)
        if proximity_factor:
            outputs.append(y_proximity)
    else:
        outputs = y

    m = Model(inputs=inp, outputs=outputs)
    m.name = 'urban_sound_classifier'
    print(m.summary())

    return m


## DATA PREPARATION

def prepare_framewise_data(train_file_idxs, test_file_idxs, embeddings,
                           target_list, sensor_list=None, proximity_list=None,
                           num_sensors=None, oversample=None, oversample_iters=1,
                           thresh_type='mean', standardize=True, pca=False,
                           pca_components=None):
    """
    Prepare inputs and targets for framewise training using training and evaluation indices.

    Parameters
    ----------
    train_file_idxs
    test_file_idxs
    embeddings
    target_list
    oversample
    oversample_iters
    thresh_type
    standardize
    pca
    pca_components

    Returns
    -------
    X_train
    y_train
    X_valid
    y_valid
    scaler

    """
    if oversample is not None and (sensor_list is not None or proximity_list is not None):
        raise ValueError('Oversampling with additional data sources is not supported')

    if oversample == 'mlsmote':
        X_train = np.array([embeddings[idx] for idx in train_file_idxs])
        y_train = np.array([target_list[idx] for idx in train_file_idxs])

        X_train_, y_train_, _ = mlsmote(X_train, y_train, oversample_iters=oversample_iters,
                                        thresh_type=thresh_type)

        X_train = []
        y_train = []
        for X, y in zip(X_train, y_train):
            X_train += list(X)
            y_train += [y for _ in range(len(X))]

        # Remove references
        X_train_ = None
        y_train_ = None
    elif oversample == 'lssmote':
        X_train = np.array([embeddings[idx] for idx in train_file_idxs])
        y_train = np.array([target_list[idx] for idx in train_file_idxs])

        X_train_, y_train_, _ = lssmote(X_train, y_train, oversample_iters=oversample_iters,
                                        thresh_type=thresh_type)

        X_train = []
        y_train = []
        for X, y in zip(X_train, y_train):
            X_train += list(X)
            y_train += [y for _ in range(len(X))]

        # Remove references
        X_train_ = None
        y_train_ = None
    elif oversample is None:

        X_train = []
        y_train = []
        for idx in train_file_idxs:
            X_ = list(embeddings[idx])
            X_train += X_
            for _ in range(len(X_)):
                y_train.append(target_list[idx])

    else:
        raise ValueError("Unknown oversample method: {}".format(oversample))

    train_idxs = np.random.permutation(len(X_train))

    X_train = np.array(X_train)[train_idxs]
    y_train = np.array(y_train)[train_idxs]

    # Add auxilliary information
    if sensor_list is not None:
        assert num_sensors is not None

        y_sensor_train = []
        for idx in train_file_idxs:
            target = sensor_list[idx]
            for _ in range(len(embeddings[idx])):
                y_sensor_train.append(target)
        y_sensor_train = np.array(y_sensor_train)[train_idxs]

        y_sensor_valid = []
        for idx in test_file_idxs:
            target = sensor_list[idx]
            for _ in range(len(embeddings[idx])):
                y_sensor_valid.append(target)
        y_sensor_valid = np.array(y_sensor_valid)

    if proximity_list is not None:
        y_proximity_train = []
        for idx in train_file_idxs:
            target = proximity_list[idx]
            for _ in range(len(embeddings[idx])):
                y_proximity_train.append(target)
        y_proximity_train = np.array(y_proximity_train)[train_idxs]

        y_proximity_valid = []
        for idx in test_file_idxs:
            target = proximity_list[idx]
            for _ in range(len(embeddings[idx])):
                y_proximity_valid.append(target)
        y_proximity_valid = np.array(y_proximity_valid)

    X_valid = []
    y_valid = []
    for idx in test_file_idxs:
        X_ = list(embeddings[idx])
        X_valid += X_
        for _ in range(len(X_)):
            y_valid.append(target_list[idx])

    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)

    # standardize
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
    else:
        scaler = None

    if pca:
        if pca_components is not None:
            pca_components = min(X_train.shape[-1], pca_components)
        pca_model = PCA(pca_components, whiten=True)

        X_train = pca_model.fit_transform(X_train)
        X_valid = pca_model.transform(X_valid)
    else:
        pca_model = None

    if sensor_list is not None or proximity_list is not None:
        y_train = {'output': y_train}
        y_valid = {'output': y_valid}

        if sensor_list is not None:
            y_train['sensor_output'] = y_sensor_train
            y_valid['sensor_output'] = y_sensor_valid

        if proximity_list is not None:
            y_train['proximity_output'] = y_proximity_train
            y_valid['proximity_output'] = y_proximity_valid

    return X_train, y_train, X_valid, y_valid, scaler, pca_model


def prepare_mil_data(train_file_idxs, valid_file_idxs, embeddings, target_list,
                     sensor_list=None, num_sensors=None, proximity_list=None,
                     standardize=True, pca=False, pca_components=None,
                     oversample=None, oversample_iters=1, thresh_type="mean"):
    """
    Prepare inputs and targets for MIL training using training and validation indices.

    Parameters
    ----------
    train_file_idxs
    valid_file_idxs
    embeddings
    target_list
    standardize
    pca
    pca_components
    oversample
    oversample_iters
    thresh_type

    Returns
    -------
    X_train
    y_train
    X_valid
    y_valid
    scaler

    """

    X_train_mil = np.array([embeddings[idx] for idx in train_file_idxs])
    X_valid_mil = np.array([embeddings[idx] for idx in valid_file_idxs])
    y_train_mil = np.array([target_list[idx] for idx in train_file_idxs])
    y_valid_mil = np.array([target_list[idx] for idx in valid_file_idxs])

    if oversample == 'mlsmote':
        X_train_mil, y_train_mil, _ = mlsmote(X_train_mil, y_train_mil, oversample_iters=oversample_iters,
                                              thresh_type=thresh_type)
    elif oversample == 'lssmote':
        X_train_mil, y_train_mil, _ = lssmote(X_train_mil, y_train_mil, oversample_iters=oversample_iters,
                                              thresh_type=thresh_type)
    elif oversample is not None:
        raise ValueError("Unknown oversample method: {}".format(oversample))

    if oversample is not None and (sensor_list is not None or proximity_list is not None):
        raise ValueError('Oversampling with additional data sources is not supported')

    # standardize
    if standardize:
        scaler = StandardScaler()
        scaler.fit(np.array([emb for emb_grp in X_train_mil for emb in emb_grp]))

        X_train_mil = [scaler.transform(emb_grp) for emb_grp in X_train_mil]
        X_valid_mil = [scaler.transform(emb_grp) for emb_grp in X_valid_mil]
    else:
        scaler = None

    # standardize
    if pca:
        if pca_components is not None:
            pca_components = min(X_train_mil[0].shape[-1], pca_components)
        pca_model = PCA(pca_components, whiten=True)
        pca_model.fit(np.array([emb for emb_grp in X_train_mil for emb in emb_grp]))

        X_train_mil = [pca_model.transform(emb_grp) for emb_grp in X_train_mil]
        X_valid_mil = [pca_model.transform(emb_grp) for emb_grp in X_valid_mil]
    else:
        pca_model = None

    train_mil_idxs = np.random.permutation(len(X_train_mil))

    X_train_mil = np.array(X_train_mil)[train_mil_idxs]
    y_train_mil = np.array(y_train_mil)[train_mil_idxs]
    X_valid_mil = np.array(X_valid_mil)


    # Add auxilliary information
    if sensor_list is not None:
        y_sensor_train_mil = []
        for idx in train_file_idxs:
            target = sensor_list[idx]
            y_sensor_train_mil.append(target)

        y_sensor_train_mil = np.array(y_sensor_train_mil)[train_mil_idxs]

        y_sensor_valid_mil = []
        for idx in valid_file_idxs:
            target = sensor_list[idx]
            y_sensor_valid_mil.append(target)
        y_sensor_valid_mil = np.array(y_sensor_valid_mil)

    if proximity_list is not None:
        y_proximity_train_mil = []
        for idx in train_file_idxs:
            target = proximity_list[idx]
            y_proximity_train_mil.append(target)
        y_proximity_train_mil = np.array(y_proximity_train_mil)[train_mil_idxs]

        y_proximity_valid_mil = []
        for idx in valid_file_idxs:
            target = proximity_list[idx]
            y_proximity_valid_mil.append(target)
        y_proximity_valid_mil = np.array(y_proximity_valid_mil)

    if sensor_list is not None or proximity_list is not None:
        y_train_mil = {'output': y_train_mil}
        y_valid_mil = {'output': y_valid_mil}

        if sensor_list is not None:
            y_train_mil['sensor_output'] = y_sensor_train_mil
            y_valid_mil['sensor_output'] = y_sensor_valid_mil

        if proximity_list is not None:
            y_train_mil['proximity_output'] = y_proximity_train_mil
            y_valid_mil['proximity_output'] = y_proximity_valid_mil

    return X_train_mil, y_train_mil, X_valid_mil, y_valid_mil, scaler, pca_model


def compute_cooccurrence_laplacian(y_train_mil):
    num_examples, num_classes = y_train_mil.shape

    A = np.zeros((num_classes, num_classes))

    for y in y_train_mil:
        active_classes = np.nonzero(y)[0]
        for idx, src_cls_idx in enumerate(active_classes):
            for dst_cls_idx in active_classes[idx+1:]:
                A[src_cls_idx, dst_cls_idx] += 1
                A[dst_cls_idx, src_cls_idx] += 1

    # Normalize the number of examples
    A /= num_examples

    D = np.diag(A.sum(axis=0))

    L = D - A
    return L


def create_graph_laplacian_loss(L):
    L = K.variable(L, dtype='float32')
    def loss(y_true, y_pred):
        y = K.expand_dims(K.sum(y_pred, axis=0))
        return K.dot(K.transpose(y), K.dot(L, y))
    return loss


## GENERIC MODEL TRAINING

def train_model(model, X_train, y_train, X_valid, y_valid, output_dir,
                loss=None, loss_weights=None, batch_size=64,
                num_epochs=100, patience=20, learning_rate=1e-4,
                optimizer='adam'):
    """
    Train a model with the given data.

    Parameters
    ----------
    model
    X_train
    y_train
    output_dir
    batch_size
    num_epochs
    patience
    learning_rate

    Returns
    -------
    history

    """

    if loss is None:
        loss = 'binary_crossentropy'
    # TODO: Update for our modified accuracy metric
    metrics = []

    os.makedirs(output_dir, exist_ok=True)

    # Set up callbacks
    cb = []
    # checkpoint
    model_weight_file = os.path.join(output_dir, 'model_best.h5')
    model.save_weights(model_weight_file)

    cb.append(keras.callbacks.ModelCheckpoint(model_weight_file,
                                              save_weights_only=True,
                                              save_best_only=True,
                                              monitor='val_loss'))
    # early stopping
    cb.append(keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=patience))

    # monitor losses
    history_csv_file = os.path.join(output_dir, 'history.csv')
    cb.append(keras.callbacks.CSVLogger(history_csv_file, append=True,
                                        separator=','))

    # Fit model
    if optimizer == 'adam':
        opt = Adam(lr=learning_rate)
    elif optimizer == 'lbfgs':
        opt = ScipyOpt(model, X_train, y_train, nb_epoch=num_epochs,
                       method='L-BFGS-B')
    else:
        raise ValueError('Invalid optimizer: {}'.format(optimizer))

    model.compile(opt, loss=loss,
                       loss_weights=loss_weights, metrics=metrics)

    if num_epochs > 0:
        history = model.fit(
            x=X_train, y=y_train, batch_size=batch_size, epochs=num_epochs,
            validation_data=(X_valid, y_valid), callbacks=cb, verbose=2)
    else:
        history = None

    return history


## MODEL TRAINING

def train_framewise(annotation_path, taxonomy_path, l3_path, raw_dir, output_dir,
                    label_mode="fine", batch_size=64, num_epochs=100,
                    patience=20, learning_rate=1e-4, hidden_layer_size=128,
                    sensor_factor=False, proximity_factor=False,
                    cooccurrence_loss=False, cooccurrence_loss_factor=1e-5,
                    num_hidden_layers=0, l2_reg=1e-5, standardize=True,
                    pca=False, pca_components=None, oversample=None,
                    oversample_iters=1, thresh_type='mean', split_path=None,
                    optimizer='adam'):
    """
    Train and evaluate a framewise MLP model.

    Parameters
    ----------
    dataset_dir
    raw_dir
    output_dir
    label_mode
    batch_size
    test_ratio
    num_epochs
    patience
    learning_rate
    hidden_layer_size
    l2_reg
    standardize
    pca
    pca_components
    oversample
    oversample_iters
    thresh_type
    split_path
    optimizer

    Returns
    -------

    """

    # Load annotations and taxonomy
    print("* Loading dataset.")
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.load(f, Loader=yaml.Loader)

    file_list = annotation_data.sort_values('audio_filename')['audio_filename'].unique().tolist()
    num_sensors = len(annotation_data.sort_values('sensor_id')['sensor_id'].unique().tolist())

    full_fine_target_labels = ["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                               for coarse_id, fine_dict in taxonomy['fine'].items()
                               for fine_id, fine_label in fine_dict.items()]
    fine_target_labels = [x for x in full_fine_target_labels
                            if x.split('_')[0].split('-')[1] != 'X']
    coarse_target_labels = ["_".join([str(k), v])
                            for k,v in taxonomy['coarse'].items()]

    print("* Preparing training data.")

    # For fine, we include incomplete labels in targets for computing the loss
    fine_target_list = get_file_targets(annotation_data, full_fine_target_labels)
    coarse_target_list = get_file_targets(annotation_data, coarse_target_labels)
    train_file_idxs, test_file_idxs = get_subset_split(annotation_data, split_path=split_path)

    if sensor_factor:
        sensor_target_list = get_file_sensor_targets(annotation_data)
    else:
        sensor_target_list = None

    if proximity_factor:
        proximity_target_list = get_file_proximity_targets(annotation_data, fine_target_labels)
    else:
        proximity_target_list = None

    if label_mode == "fine":
        target_list = fine_target_list
        labels = fine_target_labels
    elif label_mode == "coarse":
        target_list = coarse_target_list
        labels = coarse_target_labels
    else:
        raise ValueError("Invalid label mode: {}".format(label_mode))

    num_classes = len(labels)

    raw = load_raw_audio(file_list, raw_dir)

    X_train, y_train, X_valid, y_valid, scaler, pca_model \
        = prepare_framewise_data(train_file_idxs, test_file_idxs, raw,
                                 target_list, sensor_list=sensor_target_list,
                                 num_sensors=num_sensors,
                                 proximity_list=proximity_target_list,
                                 standardize=standardize,
                                 pca=pca, pca_components=pca_components,
                                 oversample=oversample,
                                 oversample_iters=oversample_iters,
                                 thresh_type=thresh_type)

    if scaler is not None:
        scaler_path = os.path.join(output_dir, 'stdizer.pkl')
        with open(scaler_path, 'wb') as f:
            pk.dump(scaler, f)

    if pca_model is not None:
        pca_path = os.path.join(output_dir, 'pca.pkl')
        with open(pca_path, 'wb') as f:
            pk.dump(pca_model, f)

    #_, emb_size = X_train.shape
    l3cnn = load_model(l3_path)

    model = construct_mlp_framewise(l3cnn, num_classes,
                                    sensor_factor=sensor_factor,
                                    num_sensors=num_sensors,
                                    proximity_factor=proximity_factor,
                                    num_proximity_classes=len(fine_target_labels),
                                    hidden_layer_size=hidden_layer_size,
                                    num_hidden_layers=num_hidden_layers,
                                    l2_reg=l2_reg)

    if label_mode == "fine":
        full_coarse_to_fine_terminal_idxs = np.cumsum(
            [len(fine_dict) for fine_dict in taxonomy['fine'].values()])
        incomplete_fine_subidxs = [len(fine_dict) - 1 if 'X' in fine_dict else None
                                   for fine_dict in taxonomy['fine'].values()]
        coarse_to_fine_end_idxs = np.cumsum([len(fine_dict) - 1 if 'X' in fine_dict else len(fine_dict)
                                             for fine_dict in taxonomy['fine'].values()])

        # Create loss function that only adds loss for fine labels for which
        # the we don't have any incomplete labels
        def masked_loss(y_true, y_pred):
            loss = None
            for coarse_idx in range(len(full_coarse_to_fine_terminal_idxs)):
                true_terminal_idx = full_coarse_to_fine_terminal_idxs[coarse_idx]
                true_incomplete_subidx = incomplete_fine_subidxs[coarse_idx]
                pred_end_idx = coarse_to_fine_end_idxs[coarse_idx]

                if coarse_idx != 0:
                    true_start_idx = full_coarse_to_fine_terminal_idxs[coarse_idx-1]
                    pred_start_idx = coarse_to_fine_end_idxs[coarse_idx-1]
                else:
                    true_start_idx = 0
                    pred_start_idx = 0

                if true_incomplete_subidx is None:
                    true_end_idx = true_terminal_idx

                    sub_true = y_true[:, true_start_idx:true_end_idx]
                    sub_pred = y_pred[:, pred_start_idx:pred_end_idx]

                else:
                    # Don't include incomplete label
                    true_end_idx = true_terminal_idx - 1
                    true_incomplete_idx = true_incomplete_subidx + true_start_idx
                    assert true_end_idx - true_start_idx == pred_end_idx - pred_start_idx
                    assert true_incomplete_idx == true_end_idx

                    # 1 if not incomplete, 0 if incomplete
                    mask = K.expand_dims(1 - y_true[:, true_incomplete_idx])

                    # Mask the target and predictions. If the mask is 0,
                    # all entries will be 0 and the BCE will be 0.
                    # This has the effect of masking the BCE for each fine
                    # label within a coarse label if an incomplete label exists
                    sub_true = y_true[:, true_start_idx:true_end_idx] * mask
                    sub_pred = y_pred[:, pred_start_idx:pred_end_idx] * mask

                if loss is not None:
                    loss += K.sum(K.binary_crossentropy(sub_true, sub_pred))
                else:
                    loss = K.sum(K.binary_crossentropy(sub_true, sub_pred))

            return loss
        loss_func = masked_loss
    else:
        loss_func = K.binary_crossentropy

    if cooccurrence_loss:
        if sensor_factor or proximity_factor:
            L = compute_cooccurrence_laplacian(y_train['output'])
        else:
            L = compute_cooccurrence_laplacian(y_train)
        laplacian_loss = create_graph_laplacian_loss(L)
        alpha = cooccurrence_loss_factor
        original_loss_func = loss_func
        def loss_with_cooccurrence(y_true, y_pred):
            return original_loss_func(y_true, y_pred) + alpha * laplacian_loss(y_true, y_pred)
        loss_func = loss_with_cooccurrence

    loss = loss_func
    loss_weights = None
    if sensor_factor or proximity_factor:
        loss = {'output': loss_func}
        loss_weights = {'output': 1.0}
        if sensor_factor:
            loss['sensor_output'] = 'categorical_crossentropy'
            loss_weights['sensor_output'] = 1.0
        if proximity_factor:
            loss['proximity_output'] = 'categorical_crossentropy'
            loss_weights['proximity_output'] = 1.0

    print("* Training model.")
    history = train_model(model, X_train, y_train, X_valid, y_valid,
                          output_dir, loss=loss, loss_weights=loss_weights,
                          batch_size=batch_size, num_epochs=num_epochs,
                          patience=patience, learning_rate=learning_rate,
                          optimizer=optimizer)

    # Reload checkpointed file
    model_weight_file = os.path.join(output_dir, 'model_best.h5')
    model.load_weights(model_weight_file)

    print("* Saving model predictions.")
    results = {}
    results['train'] = predict_framewise(raw, train_file_idxs, model,
                                         scaler=scaler, pca_model=pca_model)
    results['test'] = predict_framewise(raw, test_file_idxs, model,
                                        scaler=scaler, pca_model=pca_model)
    if history is not None:
        results['train_history'] = history.history
    else:
        results['train_history'] = {}

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    for aggregation_type, y_pred in results['test'].items():
        generate_output_file(y_pred, test_file_idxs, output_dir, file_list,
                             aggregation_type, label_mode, taxonomy)
    
    # Save Keras model in output directory
    print("* Saving Keras model.")
    keras.models.save_model(model, os.path.join(output_dir, 'mlp_ust.h5'))

def train_mil(annotation_path, taxonomy_path, emb_dir, output_dir, label_mode="fine",
              batch_size=64, num_epochs=100, patience=20, learning_rate=1e-4,
              hidden_layer_size=128, num_hidden_layers=0, sensor_factor=False,
              cooccurrence_loss=False, cooccurrence_loss_factor=1e-5,
              proximity_factor=False, l2_reg=1e-5, standardize=True,
              pca=False, pca_components=None, oversample=None, oversample_iters=1,
              thresh_type="mean", split_path=None, optimizer='adam'):
    """
    Train and evaluate a MIL MLP model.

    Parameters
    ----------
    annotation_path
    emb_dir
    output_dir
    label_mode
    batch_size
    test_ratio
    num_epochs
    patience
    learning_rate
    hidden_layer_size
    l2_reg
    standardize
    pca
    pca_components
    oversample
    oversample_iters
    thresh_type
    split_path
    optimizer

    Returns
    -------

    """
    # Load annotations and taxonomy
    print("* Loading dataset.")
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.load(f, Loader=yaml.Loader)

    file_list = annotation_data.sort_values('audio_filename')['audio_filename'].unique().tolist()
    num_sensors = len(annotation_data.sort_values('sensor_id')['sensor_id'].unique().tolist())

    full_fine_target_labels = ["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                               for coarse_id, fine_dict in taxonomy['fine'].items()
                               for fine_id, fine_label in fine_dict.items()]
    fine_target_labels = [x for x in full_fine_target_labels
                          if x.split('_')[0].split('-')[1] != 'X']
    coarse_target_labels = ["_".join([str(k), v])
                            for k,v in taxonomy['coarse'].items()]

    print("* Preparing training data.")

    # For fine, we include incomplete labels in targets for computing the loss
    fine_target_list = get_file_targets(annotation_data, full_fine_target_labels)
    coarse_target_list = get_file_targets(annotation_data, coarse_target_labels)
    train_file_idxs, test_file_idxs = get_subset_split(annotation_data, split_path=split_path)

    if sensor_factor:
        sensor_target_list = get_file_sensor_targets(annotation_data)
    else:
        sensor_target_list = None

    if proximity_factor:
        proximity_target_list = get_file_proximity_targets(annotation_data, fine_target_labels)
    else:
        proximity_target_list = None

    if label_mode == "fine":
        target_list = fine_target_list
        labels = fine_target_labels
    elif label_mode == "coarse":
        target_list = coarse_target_list
        labels = coarse_target_labels
    else:
        raise ValueError("Invalid label mode: {}".format(label_mode))

    num_classes = len(labels)

    embeddings = load_raw_audio(file_list, emb_dir)

    X_train, y_train, X_valid, y_valid, scaler, pca_model \
        = prepare_mil_data(train_file_idxs, test_file_idxs,
                           embeddings, target_list,
                           sensor_list=sensor_target_list,
                           num_sensors=num_sensors,
                           proximity_list=proximity_target_list,
                           standardize=standardize, pca=pca,
                           pca_components=pca_components,
                           oversample=oversample,
                           oversample_iters=oversample_iters,
                           thresh_type=thresh_type)

    if scaler is not None:
        scaler_path = os.path.join(output_dir, 'stdizer.pkl')
        with open(scaler_path, 'wb') as f:
            pk.dump(scaler, f)

    if pca_model is not None:
        pca_path = os.path.join(output_dir, 'pca.pkl')
        with open(pca_path, 'wb') as f:
            pk.dump(pca_model, f)

    _, num_frames, emb_size = X_train.shape

    model = construct_mlp_mil(num_frames,
                              emb_size,
                              num_classes,
                              num_hidden_layers=num_hidden_layers,
                              sensor_factor=sensor_factor,
                              num_sensors=num_sensors,
                              proximity_factor=proximity_factor,
                              num_proximity_classes=len(fine_target_labels),
                              hidden_layer_size=hidden_layer_size,
                              l2_reg=l2_reg)

    if label_mode == "fine":
        full_coarse_to_fine_terminal_idxs = np.cumsum(
            [len(fine_dict) for fine_dict in taxonomy['fine'].values()])
        incomplete_fine_subidxs = [len(fine_dict) - 1 if 'X' in fine_dict else None
                                   for fine_dict in taxonomy['fine'].values()]
        coarse_to_fine_end_idxs = np.cumsum([len(fine_dict) - 1 if 'X' in fine_dict else len(fine_dict)
                                             for fine_dict in taxonomy['fine'].values()])

        # Create loss function that only adds loss for fine labels for which
        # the we don't have any incomplete labels
        def masked_loss(y_true, y_pred):
            loss = None
            for coarse_idx in range(len(full_coarse_to_fine_terminal_idxs)):
                true_terminal_idx = full_coarse_to_fine_terminal_idxs[coarse_idx]
                true_incomplete_subidx = incomplete_fine_subidxs[coarse_idx]
                pred_end_idx = coarse_to_fine_end_idxs[coarse_idx]

                if coarse_idx != 0:
                    true_start_idx = full_coarse_to_fine_terminal_idxs[coarse_idx-1]
                    pred_start_idx = coarse_to_fine_end_idxs[coarse_idx-1]
                else:
                    true_start_idx = 0
                    pred_start_idx = 0

                if true_incomplete_subidx is None:
                    true_end_idx = true_terminal_idx

                    sub_true = y_true[:, true_start_idx:true_end_idx]
                    sub_pred = y_pred[:, pred_start_idx:pred_end_idx]

                else:
                    # Don't include incomplete label
                    true_end_idx = true_terminal_idx - 1
                    true_incomplete_idx = true_incomplete_subidx + true_start_idx
                    assert true_end_idx - true_start_idx == pred_end_idx - pred_start_idx
                    assert true_incomplete_idx == true_end_idx

                    # 1 if not incomplete, 0 if incomplete
                    mask = K.expand_dims(1 - y_true[:, true_incomplete_idx])

                    # Mask the target and predictions. If the mask is 0,
                    # all entries will be 0 and the BCE will be 0.
                    # This has the effect of masking the BCE for each fine
                    # label within a coarse label if an incomplete label exists
                    sub_true = y_true[:, true_start_idx:true_end_idx] * mask
                    sub_pred = y_pred[:, pred_start_idx:pred_end_idx] * mask

                if loss is not None:
                    loss += K.sum(K.binary_crossentropy(sub_true, sub_pred))
                else:
                    loss = K.sum(K.binary_crossentropy(sub_true, sub_pred))

            return loss
        loss_func = masked_loss
    else:
        loss_func = K.binary_crossentropy

    if cooccurrence_loss:
        if sensor_factor or proximity_factor:
            L = compute_cooccurrence_laplacian(y_train['output'])
        else:
            L = compute_cooccurrence_laplacian(y_train)
        laplacian_loss = create_graph_laplacian_loss(L)
        alpha = cooccurrence_loss_factor
        original_loss_func = loss_func
        def loss_with_cooccurrence(y_true, y_pred):
            return original_loss_func(y_true, y_pred) + alpha * laplacian_loss(y_true, y_pred)
        loss_func = loss_with_cooccurrence

    loss = loss_func
    loss_weights = None
    if sensor_factor or proximity_factor:
        loss = {'output': loss_func}
        loss_weights = {'output': 1.0}
        if sensor_factor:
            loss['sensor_output'] = 'categorical_crossentropy'
            loss_weights['sensor_output'] = 1.0
        if proximity_factor:
            loss['proximity_output'] = 'categorical_crossentropy'
            loss_weights['proximity_output'] = 1.0

    print("* Training model.")
    history = train_model(model, X_train, y_train, X_valid, y_valid,
                          output_dir, loss=loss, loss_weights=loss_weights,
                          batch_size=batch_size, num_epochs=num_epochs,
                          patience=patience, learning_rate=learning_rate,
                          optimizer=optimizer)

    # Reload checkpointed file
    model_weight_file = os.path.join(output_dir, 'model_best.h5')
    model.load_weights(model_weight_file)

    print("* Saving model predictions.")
    results = {}
    results['train'] = predict_mil(embeddings, train_file_idxs, model,
                                   scaler=scaler, pca_model=pca_model)
    results['test'] = predict_mil(embeddings, test_file_idxs, model,
                                  scaler=scaler, pca_model=pca_model)
    results['train_history'] = history.history

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    generate_output_file(results['test'], test_file_idxs, output_dir, file_list,
                         "", label_mode, taxonomy)


## MODEL EVALUATION

def predict_framewise(embeddings, test_file_idxs, model, scaler=None, pca_model=None):
    """
    Evaluate the output of a framewise classification model.

    Parameters
    ----------
    embeddings
    test_file_idxs
    model
    scaler
    pca_model

    Returns
    -------
    results
    """
    y_pred_max = []
    y_pred_mean = []
    y_pred_softmax = []

    for idx in test_file_idxs:
        if scaler is None:
            X_ = np.array(embeddings[idx])
        else:
            X_ = np.array(scaler.transform(embeddings[idx]))

        if pca_model is not None:
            X_ = pca_model.transform(X_)

        pred_frame = model.predict(X_)
        # Discard auxilliary predictions
        if type(pred_frame) == list:
            pred_frame = pred_frame[0]

        y_pred_max.append(pred_frame.max(axis=0).tolist())
        y_pred_mean.append(pred_frame.mean(axis=0).tolist())
        y_pred_softmax.append(((softmax(pred_frame, axis=0) * pred_frame).sum(axis=0)).tolist())

    results = {
        'max': y_pred_max,
        'mean': y_pred_mean,
        'softmax': y_pred_softmax
    }

    return results


def predict_mil(embeddings, test_file_idxs, model, scaler=None, pca_model=None):
    """
    Evaluate the output of a MIL classification model.

    Parameters
    ----------
    embeddings
    test_file_idxs
    model
    scaler
    pca_model

    Returns
    -------
    results
    """
    if scaler is None:
        X = np.array([embeddings[idx] for idx in test_file_idxs])
    else:
        X = np.array([scaler.transform(embeddings[idx]) for idx in test_file_idxs])

    if pca_model is not None:
        X = np.array([pca_model.transform(ex) for ex in X])

    pred = model.predict(X)

    # Discard auxilliary predictions
    if type(pred) == list:
        pred = pred[0]

    return pred.tolist()


def generate_output_file(y_pred, test_file_idxs, results_dir, file_list,
                         aggregation_type, label_mode, taxonomy):
    """
    Write the output file containing model predictions

    Parameters
    ----------
    y_pred
    test_file_idxs
    results_dir
    file_list
    aggregation_type
    label_mode
    taxonomy

    Returns
    -------

    """
    if aggregation_type:
        output_path = os.path.join(results_dir, "output_{}.csv".format(aggregation_type))
    else:
        output_path = os.path.join(results_dir, "output.csv")
    test_file_list = [file_list[idx] for idx in test_file_idxs]

    coarse_fine_labels = [["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                             for fine_id, fine_label in fine_dict.items()]
                           for coarse_id, fine_dict in taxonomy['fine'].items()]

    full_fine_target_labels = [fine_label for fine_list in coarse_fine_labels
                                          for fine_label in fine_list]
    coarse_target_labels = ["_".join([str(k), v])
                            for k,v in taxonomy['coarse'].items()]


    with open(output_path, 'w') as f:
        csvwriter = csv.writer(f)

        # Write fields
        fields = ["audio_filename"] + full_fine_target_labels + coarse_target_labels
        csvwriter.writerow(fields)

        # Write results for each file to CSV
        for filename, y, in zip(test_file_list, y_pred):
            row = [filename]

            if label_mode == "fine":
                fine_values = []
                coarse_values = [0 for _ in range(len(coarse_target_labels))]
                coarse_idx = 0
                fine_idx = 0
                for coarse_label, fine_label_list in zip(coarse_target_labels,
                                                         coarse_fine_labels):
                    for fine_label in fine_label_list:
                        if 'X' in fine_label.split('_')[0].split('-')[1]:
                            # Put a 0 for other, since the baseline doesn't
                            # account for it
                            fine_values.append(0.0)
                            continue

                        # Append the next fine prediction
                        fine_values.append(y[fine_idx])

                        # Add coarse level labels corresponding to fine level
                        # predictions. Obtain by taking the maximum from the
                        # fine level labels
                        coarse_values[coarse_idx] = max(coarse_values[coarse_idx],
                                                        y[fine_idx])
                        fine_idx += 1
                    coarse_idx += 1

                row += fine_values + coarse_values

            else:
                # Add placeholder values for fine level
                row += [0.0 for _ in range(len(full_fine_target_labels))]
                # Add coarse level labels
                row += list(y)

            csvwriter.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path")
    parser.add_argument("taxonomy_path")
    parser.add_argument("l3_path")
    parser.add_argument("raw_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("exp_id", type=str)

    parser.add_argument("--hidden_layer_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--sensor_factor", action='store_true')
    parser.add_argument("--proximity_factor", action='store_true')
    parser.add_argument("--no_standardize", action='store_true')
    parser.add_argument("--cooccurrence_loss", action='store_true')
    parser.add_argument("--cooccurrence_loss_factor", type=float, default=1e-5)
    parser.add_argument("--pca", action='store_true')
    parser.add_argument("--pca_components", type=int)
    parser.add_argument("--label_mode", type=str, choices=["fine", "coarse"],
                        default='fine')
    parser.add_argument("--oversample", type=str, choices=["mlsmote", "lssmote"])
    parser.add_argument("--oversample_iters", type=int, default=1)
    parser.add_argument("--thresh_type", type=str, default="mean",
                        choices=["mean"] + ["percentile_{}".format(i) for i in range(1,100)])
    parser.add_argument("--target_mode", type=str, choices=["framewise", "mil"],
                        default='framewise')
    parser.add_argument("--no_timestamp", action='store_true')
    parser.add_argument("--split_path", type=str)
    parser.add_argument("--optimizer", type=str, choices=["adam", "lbfgs"])

    args = parser.parse_args()

    # save args to disk
    if args.no_timestamp:
        out_dir = os.path.join(args.output_dir, args.exp_id)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        out_dir = os.path.join(args.output_dir, args.exp_id, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    kwarg_file = os.path.join(out_dir, "hyper_params.json")
    with open(kwarg_file, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Print configuration
    print("Configuration:")
    pprint.pprint(vars(args))
    print()

    if args.target_mode == 'mil':
        raise NotImplementedError("End-to-end pipeline not yet supported with MIL yet.")
        # train_mil(args.annotation_path,
        #           args.taxonomy_path,
        #           args.raw_dir,
        #           out_dir,
        #           label_mode=args.label_mode,
        #           batch_size=args.batch_size,
        #           num_epochs=args.num_epochs,
        #           patience=args.patience,
        #           learning_rate=args.learning_rate,
        #           hidden_layer_size=args.hidden_layer_size,
        #           num_hidden_layers=args.num_hidden_layers,
        #           sensor_factor=args.sensor_factor,
        #           proximity_factor=args.proximity_factor,
        #           cooccurrence_loss=args.cooccurrence_loss,
        #           cooccurrence_loss_factor=args.cooccurrence_loss_factor,
        #           l2_reg=args.l2_reg,
        #           standardize=(not args.no_standardize),
        #           pca=args.pca,
        #           pca_components=args.pca_components,
        #           oversample=args.oversample,
        #           oversample_iters=args.oversample_iters,
        #           thresh_type=args.thresh_type,
        #           split_path=args.split_path,
        #           optimizer=args.optimizer)
    elif args.target_mode == 'framewise':
        train_framewise(args.annotation_path,
                        args.taxonomy_path,
                        args.l3_path,
                        args.emb_dir,
                        out_dir,
                        label_mode=args.label_mode,
                        batch_size=args.batch_size,
                        num_epochs=args.num_epochs,
                        patience=args.patience,
                        learning_rate=args.learning_rate,
                        hidden_layer_size=args.hidden_layer_size,
                        num_hidden_layers=args.num_hidden_layers,
                        sensor_factor=args.sensor_factor,
                        proximity_factor=args.proximity_factor,
                        cooccurrence_loss=args.cooccurrence_loss,
                        cooccurrence_loss_factor=args.cooccurrence_loss_factor,
                        l2_reg=args.l2_reg,
                        standardize=False, #(not args.no_standardize),
                        pca=args.pca,
                        pca_components=args.pca_components,
                        oversample=args.oversample,
                        oversample_iters=args.oversample_iters,
                        thresh_type=args.thresh_type,
                        split_path=args.split_path,
                        optimizer=args.optimizer)
