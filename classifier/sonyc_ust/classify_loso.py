import argparse
import datetime
import json
import pickle as pk
import gzip
import os
import sys
import oyaml as yaml
import numpy as np
import pandas as pd

import keras
from keras.layers import Input, Dense, TimeDistributed
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam
import keras.backend as K
from autopool import AutoPool1D
from sklearn.preprocessing import StandardScaler

from balancing import mlsmote, lssmote
from classify import load_embeddings, construct_mlp_framewise, construct_mlp_mil, prepare_framewise_data, prepare_mil_data, train_model, predict_mil, predict_framewise, generate_output_file


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
                if int(row['annotator_id']) <= 0:
                    # Skip verified annotations
                    continue

                count += row[label + '_presence']

            if count > 0:
                target.append(1.0)
            else:
                target.append(0.0)

        target_list.append(target)

    return np.array(target_list)

## MODEL TRAINING


def train_framewise_loso(annotation_path, taxonomy_path, emb_dir, output_dir,
                         label_mode="fine", batch_size=64, num_epochs=100, patience=20,
                         learning_rate=1e-4, hidden_layer_size=128, num_hidden_layers=0,
                         l2_reg=1e-5, standardize=True, pca=False, pca_components=None,
                         oversample=None, oversample_iters=1, thresh_type="mean"):
    """
    Train and evaluate a framewise MLP model.

    Parameters
    ----------
    annotation_path
    taxonomy_path
    emb_dir
    output_dir
    label_mode
    batch_size
    num_epochs
    patience
    learning_rate
    hidden_layer_size
    num_hidden_layers
    l2_reg
    standardize
    pca
    pca_components
    oversample
    oversample_iters
    thresh_type

    Returns
    -------

    """
    print("* Loading dataset.")
    sys.stdout.flush()
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.load(f, Loader=yaml.Loader)

    file_list = annotation_data.sort_values('audio_filename')['audio_filename'].unique().tolist()

    full_fine_target_labels = ["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                               for coarse_id, fine_dict in taxonomy['fine'].items()
                               for fine_id, fine_label in fine_dict.items()]
    fine_target_labels = [x for x in full_fine_target_labels
                            if x.split('_')[0].split('-')[1] != 'X']
    coarse_target_labels = ["_".join([str(k), v])
                            for k,v in taxonomy['coarse'].items()]

    sensors_to_idxs = {}
    for idx, (_, row) in enumerate(annotation_data[['audio_filename', 'sensor_id']].drop_duplicates().iterrows()):
        sensor_id = row['sensor_id']
        if sensor_id not in sensors_to_idxs:
            sensors_to_idxs[sensor_id] = []

        sensors_to_idxs[sensor_id].append(idx)
    print("* Preparing training data.")
    sys.stdout.flush()

    # For fine, we include incomplete labels in targets for computing the loss
    fine_target_list = get_file_targets(annotation_data, full_fine_target_labels)
    coarse_target_list = get_file_targets(annotation_data, coarse_target_labels)

    if label_mode == "fine":
        target_list = fine_target_list
        labels = fine_target_labels
    elif label_mode == "coarse":
        target_list = coarse_target_list
        labels = coarse_target_labels
    else:
        raise ValueError("Invalid label mode: {}".format(label_mode))

    num_classes = len(labels)

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
        loss_func = None

    results = {}
    embeddings = load_embeddings(file_list, emb_dir)
    for sensor_id in sensors_to_idxs.keys():
        train_file_idxs = [idx for sid in sensors_to_idxs.keys()
                               for idx in sensors_to_idxs[sid] if sid != sensor_id]
        valid_file_idxs = sensors_to_idxs[sensor_id]

        X_train, y_train, X_valid, y_valid, scaler, pca_model \
            = prepare_framewise_data(train_file_idxs, valid_file_idxs,
                                     embeddings, target_list, standardize=standardize,
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

        _, emb_size = X_train.shape

        model = construct_mlp_framewise(emb_size, num_classes,
                                        hidden_layer_size=hidden_layer_size,
                                        num_hidden_layers=num_hidden_layers,
                                        l2_reg=l2_reg)

        run_dir = os.path.join(output_dir, str(sensor_id))
        os.makedirs(run_dir, exist_ok=True)

        print("* Training model.")
        sys.stdout.flush()
        history = train_model(model, X_train, y_train, X_valid, y_valid,
                              run_dir, loss=loss_func, batch_size=batch_size,
                              num_epochs=num_epochs, patience=patience,
                              learning_rate=learning_rate)

        # Reload checkpointed file
        model_weight_file = os.path.join(run_dir, 'model_best.h5')
        model.load_weights(model_weight_file)

        print("* Saving model predictions.")
        sys.stdout.flush()
        results[sensor_id] = {}
        results[sensor_id]['train'] = predict_framewise(embeddings, train_file_idxs,
                                                        model, scaler=scaler,
                                                        pca_model=pca_model)
        results[sensor_id]['test'] = predict_framewise(embeddings, valid_file_idxs, model,
                                                       scaler=scaler, pca_model=pca_model)
        results[sensor_id]['train_history'] = history.history

        for aggregation_type, y_pred in results[sensor_id]['test'].items():
            generate_output_file(y_pred, valid_file_idxs, run_dir, file_list,
                                 aggregation_type, label_mode, taxonomy)

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def train_mil_loso(annotation_path, taxonomy_path, emb_dir, output_dir, label_mode="fine",
                   batch_size=64, num_epochs=100, patience=20, learning_rate=1e-4,
                   hidden_layer_size=128, num_hidden_layers=0, l2_reg=1e-5,
                   standardize=True, pca=False, pca_components=None, oversample=None,
                   oversample_iters=1, thresh_type="mean"):
    """
    Train and evaluate a MIL MLP model.

    Parameters
    ----------
    annotation_path
    taxonomy_path
    emb_dir
    output_dir
    label_mode
    batch_size
    num_epochs
    patience
    learning_rate
    hidden_layer_size
    num_hidden_layers
    l2_reg
    standardize
    pca
    pca_components
    oversample
    oversample_iters
    thresh_type

    Returns
    -------

    """

    # Load annotations and taxonomy
    print("* Loading dataset.")
    sys.stdout.flush()
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.load(f, Loader=yaml.Loader)

    file_list = annotation_data.sort_values('audio_filename')['audio_filename'].unique().tolist()

    full_fine_target_labels = ["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                               for coarse_id, fine_dict in taxonomy['fine'].items()
                               for fine_id, fine_label in fine_dict.items()]
    fine_target_labels = [x for x in full_fine_target_labels
                          if x.split('_')[0].split('-')[1] != 'X']
    coarse_target_labels = ["_".join([str(k), v])
                            for k,v in taxonomy['coarse'].items()]

    sensors_to_idxs = {}
    for idx, (_, row) in enumerate(annotation_data[['audio_filename', 'sensor_id']].drop_duplicates().sort_values('audio_filename').iterrows()):
        sensor_id = row['sensor_id']
        if sensor_id not in sensors_to_idxs:
            sensors_to_idxs[sensor_id] = []

        sensors_to_idxs[sensor_id].append(idx)


    print("* Preparing training data.")
    sys.stdout.flush()

    # For fine, we include incomplete labels in targets for computing the loss
    fine_target_list = get_file_targets(annotation_data, full_fine_target_labels)
    coarse_target_list = get_file_targets(annotation_data, coarse_target_labels)

    if label_mode == "fine":
        target_list = fine_target_list
        labels = fine_target_labels
    elif label_mode == "coarse":
        target_list = coarse_target_list
        labels = coarse_target_labels
    else:
        raise ValueError("Invalid label mode: {}".format(label_mode))

    num_classes = len(labels)

    embeddings = load_embeddings(file_list, emb_dir)

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
        loss_func = None


    results = {}

    for sensor_id in sensors_to_idxs.keys():
        train_file_idxs = [idx for sid in sensors_to_idxs.keys()
                               for idx in sensors_to_idxs[sid] if sid != sensor_id]
        valid_file_idxs = sensors_to_idxs[sensor_id]

        X_train, y_train, X_valid, y_valid, scaler, pca_model \
            = prepare_mil_data(train_file_idxs, valid_file_idxs, embeddings,
                               target_list, standardize=standardize,
                               pca=pca, pca_components=pca_components,
                               oversample=oversample, oversample_iters=oversample_iters,
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
                                  hidden_layer_size=hidden_layer_size,
                                  l2_reg=l2_reg)

        run_dir = os.path.join(output_dir, str(sensor_id))
        os.makedirs(run_dir, exist_ok=True)

        print("* Training model.")
        sys.stdout.flush()
        history = train_model(model, X_train, y_train, X_valid, y_valid,
                              run_dir, loss=loss_func, batch_size=batch_size,
                              num_epochs=num_epochs, patience=patience,
                              learning_rate=learning_rate)

        # Reload checkpointed file
        model_weight_file = os.path.join(run_dir, 'model_best.h5')
        model.load_weights(model_weight_file)

        print("* Saving model predictions.")
        sys.stdout.flush()
        results[sensor_id] = {}
        results[sensor_id]['train'] = predict_mil(embeddings, train_file_idxs, model,
                                             scaler=scaler, pca_model=pca_model)
        results[sensor_id]['test'] = predict_mil(embeddings, valid_file_idxs, model,
                                            scaler=scaler, pca_model=pca_model)
        results[sensor_id]['train_history'] = history.history

        generate_output_file(results[sensor_id]['test'], valid_file_idxs, run_dir, file_list,
                             "", label_mode, taxonomy)

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path")
    parser.add_argument("taxonomy_path")
    parser.add_argument("emb_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("exp_id", type=str)

    parser.add_argument("--hidden_layer_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--l2_reg", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--no_standardize", action='store_true')
    parser.add_argument("--pca", action='store_true')
    parser.add_argument("--pca_components", type=int)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--label_mode", type=str, choices=["fine", "coarse"],
                        default='fine')
    parser.add_argument("--oversample", type=str, choices=["mlsmote", "lssmote"])
    parser.add_argument("--oversample_iters", type=int, default=1)
    parser.add_argument("--thresh_type", type=str, default="mean",
                        choices=["mean"] + ["percentile_{}".format(i) for i in range(1,100)])
    parser.add_argument("--target_mode", type=str, choices=["framewise", "mil"],
                        default='framewise')
    parser.add_argument("--no_timestamp", action='store_true')


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

    if args.target_mode == 'mil':
        train_mil_loso(args.annotation_path,
                       args.taxonomy_path,
                       args.emb_dir,
                       out_dir,
                       label_mode=args.label_mode,
                       batch_size=args.batch_size,
                       num_epochs=args.num_epochs,
                       patience=args.patience,
                       learning_rate=args.learning_rate,
                       hidden_layer_size=args.hidden_layer_size,
                       num_hidden_layers=args.num_hidden_layers,
                       l2_reg=args.l2_reg,
                       standardize=(not args.no_standardize),
                       pca=args.pca,
                       pca_components=args.pca_components,
                       oversample=args.oversample,
                       oversample_iters=args.oversample_iters,
                       thresh_type=args.thresh_type)
    elif args.target_mode == 'framewise':
        train_framewise_loso(args.annotation_path,
                             args.taxonomy_path,
                             args.emb_dir,
                             out_dir,
                             label_mode=args.label_mode,
                             batch_size=args.batch_size,
                             num_epochs=args.num_epochs,
                             patience=args.patience,
                             learning_rate=args.learning_rate,
                             hidden_layer_size=args.hidden_layer_size,
                             num_hidden_layers=args.num_hidden_layers,
                             l2_reg=args.l2_reg,
                             standardize=(not args.no_standardize),
                             pca=args.pca,
                             pca_components=args.pca_components,
                             oversample=args.oversample,
                             oversample_iters=args.oversample_iters,
                             thresh_type=args.thresh_type)
