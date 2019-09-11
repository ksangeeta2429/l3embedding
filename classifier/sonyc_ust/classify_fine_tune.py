import argparse
import datetime
import json
import os
import pprint
import numpy as np
import pandas as pd
import oyaml as yaml
import librosa
import pescador

import keras
from keras.layers import Input, Dense, TimeDistributed, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
import keras.backend as K
from autopool import AutoPool1D

from classify import (get_subset_split, get_file_targets,
                      get_file_sensor_targets, get_file_proximity_targets,
                      softmax, generate_output_file)

import openl3.models


def load_audio(filename, audio_dir, hop_size=0.1):
    """
    Load saved audio_list from an embedding directory

    Parameters
    ----------
    file_list
    audio_dir

    Returns
    -------
    audio_list

    """

    audio_path = os.path.join(audio_dir, 'train', filename)
    if not os.path.exists(audio_path):
        audio_path = os.path.join(audio_dir, 'validate', filename)
        if not os.path.exists(audio_path):
            audio_path = os.path.join(audio_dir, 'test', filename)

    audio, _ = librosa.load(audio_path, sr=48000)

    pad_length = 48000 * 10 - audio.shape[0]
    if pad_length > 0:
        audio = np.pad(audio, (0, pad_length), mode='constant')
    elif pad_length < 0:
        audio = audio[:48000*10]

    frame_length = 48000
    hop_length = int(hop_size * 48000)
    audio_frames = librosa.util.frame(audio, frame_length, hop_length).T
    # Add channel dim for Keras
    audio_frames = audio_frames[:, np.newaxis, :]

    return audio_frames


## GENERIC MODEL TRAINING
def train_model(model, train_gen, valid_gen, output_dir,
                steps_per_epoch, valid_steps_per_epoch,
                loss=None, loss_weights=None,
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
    else:
        raise ValueError('Invalid optimizer: {}'.format(optimizer))

    model.compile(opt, loss=loss,
                  loss_weights=loss_weights, metrics=metrics)

    if num_epochs > 0:
        history = model.fit_generator(
            train_gen, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
            validation_steps=valid_steps_per_epoch,
            validation_data=valid_gen, callbacks=cb, verbose=2)
    else:
        history = None

    return history


## MODEL CONSTRUCTION
def configure_trainable(model, num_trainable_layers):
    active_layers = 0
    for layer in reversed(model.layers):
        if len(layer._trainable_weights) > 0 and active_layers < num_trainable_layers:
            layer.trainable = True
            active_layers += 1
        else:
            layer.trainable = False


def construct_mil_model(num_frames, num_classes, sensor_factor=True,
                        num_sensors=None, proximity_factor=True, num_proximity_classes=None,
                        num_trainable_openl3_layers=2,
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
    input_repr = 'mel256'
    content_type = 'music'
    embedding_size = 512
    asr = 48000

    # Input layer
    inp = Input(shape=(num_frames, 1, asr), dtype='float32', name='input')

    emb_model = openl3.models.load_audio_embedding_model(input_repr, content_type, embedding_size)
    configure_trainable(emb_model, num_trainable_openl3_layers)

    y = TimeDistributed(emb_model, input_shape=(num_frames, 1, asr))(inp)

    # Add hidden layers
    repr_size = embedding_size
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
def data_generator(file_idxs, file_list, audio_dir, target_list, hop_size=0.1,
                   sensor_list=None, proximity_list=None):
    """
    Prepare inputs and targets for MIL training using training and validation indices.

    Parameters
    ----------
    train_file_idxs
    valid_file_idxs
    audio_list
    target_list

    Returns
    -------
    X_train
    y_train
    X_valid
    y_valid
    scaler

    """

    file_idxs = np.array(file_idxs)
    np.random.shuffle(file_idxs)

    for idx in file_idxs:
        audio_filename = file_list[idx]
        audio_frames = load_audio(audio_filename, audio_dir, hop_size=hop_size)

        target = target_list[idx]

        sample_dict = {
            'audio_frames': audio_frames,
            'target': target,
        }

        # Add auxilliary information
        if sensor_list is not None:
            sensor_target = sensor_list[idx]
            sample_dict['sensor_target'] = sensor_target

        if proximity_list is not None:
            proximity_target = proximity_list[idx]
            sample_dict['proximity_target'] = proximity_target

        yield sample_dict


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


## MODEL TRAINING
def train_mil(annotation_path, taxonomy_path, audio_dir, output_dir,
              hop_size=0.1, label_mode="fine",
              batch_size=64, num_epochs=100, patience=20, learning_rate=1e-4,
              hidden_layer_size=128, num_hidden_layers=0, sensor_factor=False,
              cooccurrence_loss=False, cooccurrence_loss_factor=1e-5,
              num_trainable_openl3_layers=2,
              proximity_factor=False, l2_reg=1e-5, optimizer='adam'):
    """
    Train and evaluate a MIL MLP model.

    Parameters
    ----------
    annotation_path
    audio_dir
    output_dir
    label_mode
    batch_size
    test_ratio
    num_epochs
    patience
    learning_rate
    hidden_layer_size
    l2_reg
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
    train_file_idxs, test_file_idxs = get_subset_split(annotation_data)

    target_name_list = ['target']

    if sensor_factor:
        sensor_target_list = get_file_sensor_targets(annotation_data)
        target_name_list.append('sensor_target')
    else:
        sensor_target_list = None

    if proximity_factor:
        proximity_target_list = get_file_proximity_targets(annotation_data, fine_target_labels)
        target_name_list.append('proximity_target')
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

    train_stream = pescador.Streamer(data_generator, train_file_idxs, file_list,
                                     audio_dir, target_list, hop_size=hop_size,
                                     sensor_list=sensor_target_list,
                                     proximity_list=proximity_target_list).cycle()
    train_gen = pescador.maps.buffer_stream(train_stream, batch_size,
                                            partial=True)
    train_gen = pescador.maps.keras_tuples(train_gen, inputs='audio_frames',
                                           outputs=target_name_list)

    steps_per_epoch = int(np.ceil(len(train_file_idxs) / batch_size))

    # Create valid batches once
    valid_stream = pescador.Streamer(data_generator, test_file_idxs, file_list,
                                     audio_dir, target_list, hop_size=hop_size,
                                     sensor_list=sensor_target_list,
                                     proximity_list=proximity_target_list).cycle()
    valid_gen = pescador.maps.buffer_stream(valid_stream, batch_size,
                                            partial=True)
    valid_gen = pescador.maps.keras_tuples(valid_gen, inputs='audio_frames',
                                           outputs=target_name_list)
    valid_steps_per_epoch = int(np.ceil(len(test_file_idxs) / batch_size))

    hop_length = int(hop_size * 48000)
    num_frames = int(48000 * (10 - 1) // hop_length + 1)

    model = construct_mil_model(num_frames,
                                num_classes,
                                num_hidden_layers=num_hidden_layers,
                                sensor_factor=sensor_factor,
                                num_sensors=num_sensors,
                                num_trainable_openl3_layers=num_trainable_openl3_layers,
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
        train_target_arr = np.array([target_list[idx] for idx in train_file_idxs])

        if sensor_factor or proximity_factor:
            L = compute_cooccurrence_laplacian(train_target_arr)
        else:
            L = compute_cooccurrence_laplacian(train_target_arr)
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
    history = train_model(model, train_gen, valid_gen,
                          output_dir, steps_per_epoch, valid_steps_per_epoch,
                          loss=loss, loss_weights=loss_weights,
                          num_epochs=num_epochs, patience=patience,
                          learning_rate=learning_rate, optimizer=optimizer)

    # Reload checkpointed file
    model_weight_file = os.path.join(output_dir, 'model_best.h5')
    model.load_weights(model_weight_file)

    print("* Saving model predictions.")
    results = {}
    results['train'] = predict_mil(file_list, train_file_idxs, model, audio_dir, hop_size)
    results['test'] = predict_mil(file_list, test_file_idxs, model, audio_dir, hop_size)
    results['train_history'] = history.history

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    generate_output_file(results['test'], test_file_idxs, output_dir, file_list,
                         "", label_mode, taxonomy)


## MODEL EVALUATION

def predict_mil(file_list, test_file_idxs, model, audio_dir, hop_size):
    """
    Evaluate the output of a MIL classification model.

    Parameters
    ----------
    audio_list
    test_file_idxs
    model

    Returns
    -------
    results
    """
    X = np.array([load_audio(file_list[idx], audio_dir, hop_size=hop_size)
                  for idx in test_file_idxs])
    pred = model.predict(X)

    # Discard auxilliary predictions
    if type(pred) == list:
        pred = pred[0]

    return pred.tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path")
    parser.add_argument("taxonomy_path")
    parser.add_argument("audio_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("exp_id", type=str)

    parser.add_argument("--hidden_layer_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hop_size", type=float, default=0.1)
    parser.add_argument("--l2_reg", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--sensor_factor", action='store_true')
    parser.add_argument("--proximity_factor", action='store_true')
    parser.add_argument("--cooccurrence_loss", action='store_true')
    parser.add_argument("--cooccurrence_loss_factor", type=float, default=1e-5)
    parser.add_argument("--num_trainable_openl3_layers", type=int, default=0)
    parser.add_argument("--label_mode", type=str, choices=["fine", "coarse"],
                        default='fine')
    parser.add_argument("--no_timestamp", action='store_true')
    parser.add_argument("--optimizer", type=str, choices=["adam"], default='adam')

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

    train_mil(args.annotation_path,
              args.taxonomy_path,
              args.audio_dir,
              out_dir,
              hop_size=args.hop_size,
              label_mode=args.label_mode,
              batch_size=args.batch_size,
              num_epochs=args.num_epochs,
              patience=args.patience,
              learning_rate=args.learning_rate,
              hidden_layer_size=args.hidden_layer_size,
              num_hidden_layers=args.num_hidden_layers,
              num_trainable_openl3_layers=args.num_trainable_openl3_layers,
              sensor_factor=args.sensor_factor,
              proximity_factor=args.proximity_factor,
              cooccurrence_loss=args.cooccurrence_loss,
              cooccurrence_loss_factor=args.cooccurrence_loss_factor,
              l2_reg=args.l2_reg,
              optimizer=args.optimizer)
