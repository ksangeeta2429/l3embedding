import argparse
import datetime
import json
import pickle as pk
import os
import sys
import csv
import numpy as np
import pandas as pd

from classify import load_embeddings, construct_mlp_framewise, prepare_framewise_data, train_model, predict_framewise


def generate_output_file(y_pred, test_file_idxs, results_dir, file_list,
                         aggregation_type):
    """
    Write the output file containing model predictions

    Parameters
    ----------
    y_pred
    test_file_idxs
    results_dir
    file_list
    aggregation_type

    Returns
    -------

    """
    if aggregation_type:
        output_path = os.path.join(results_dir, "output_{}.csv".format(aggregation_type))
    else:
        output_path = os.path.join(results_dir, "output.csv")
    test_file_list = [file_list[idx] for idx in test_file_idxs]

    with open(output_path, 'w') as f:
        csvwriter = csv.writer(f)

        # Write fields
        fields = ["audio_filename", ] + ['class{}'.format(i) for i in range(len(y_pred[0]))]
        csvwriter.writerow(fields)

        # Write results for each file to CSV
        for filename, y, in zip(test_file_list, y_pred):
            row = [filename, ] + list(y)
            csvwriter.writerow(row)


def train_framewise_sensor(split_path, emb_dir, output_dir,
                           batch_size=64, num_epochs=100, patience=20,
                           learning_rate=1e-4, hidden_layer_size=128, num_hidden_layers=0,
                           l2_reg=1e-5, standardize=True, pca=False, pca_components=None,
                           oversample=None, oversample_iters=1, thresh_type="mean", overwrite=True):
    """
    Train and evaluate a framewise MLP model.

    Parameters
    ----------
    split_path
    emb_dir
    output_dir
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
    splits_df = pd.read_csv(split_path)
    file_list = splits_df['filename'].values
    target_list = splits_df['sensor_id'].values
    sensor_ids = list(np.sort(splits_df['sensor_id'].unique()))
    _target_list = np.zeros([len(target_list), len(sensor_ids)])
    for i in  range(target_list.shape[0]):
        _target_list[i, sensor_ids.index(target_list[i])] = 1
    target_list = _target_list

    print("* Preparing training data.")
    sys.stdout.flush()

    num_classes = len(sensor_ids)

    loss_func = None

    results = {}
    embeddings = load_embeddings(file_list, emb_dir)
    for split in splits_df['split'].unique():
        run_dir = os.path.join(output_dir, str(split))

        if not overwrite:
            if os.path.exists(os.path.join(run_dir, "output_mean.csv")) or \
               os.path.exists(os.path.join(run_dir, "output.csv")):
                print("Found existing output file")
                continue

        train_file_idxs = np.nonzero((splits_df['split'] != split).values)[0]
        valid_file_idxs = np.nonzero((splits_df['split'] == split).values)[0]

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
                                        l2_reg=l2_reg, activation='softmax')

        os.makedirs(run_dir, exist_ok=True)

        print("* Training model.")
        sys.stdout.flush()
        history = train_model(model, X_train, y_train, X_valid, y_valid,
                              run_dir, loss='categorical_crossentropy', batch_size=batch_size,
                              num_epochs=num_epochs, patience=patience,
                              learning_rate=learning_rate)

        # Reload checkpointed file
        model_weight_file = os.path.join(run_dir, 'model_best.h5')
        model.load_weights(model_weight_file)

        print("* Saving model predictions.")
        sys.stdout.flush()
        results[str(split)] = {}
        results[str(split)]['train'] = predict_framewise(embeddings, train_file_idxs,
                                                         model, scaler=scaler,
                                                         pca_model=pca_model)
        results[str(split)]['test'] = predict_framewise(embeddings, valid_file_idxs, model,
                                                        scaler=scaler, pca_model=pca_model)
        results[str(split)]['train_history'] = history.history

        for aggregation_type, y_pred in results[str(split)]['test'].items():
            generate_output_file(y_pred, valid_file_idxs, run_dir, file_list,
                                 aggregation_type)

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("split_path")
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
    parser.add_argument("--no_overwrite", action="store_true")
    parser.add_argument("--oversample", type=str, choices=["mlsmote", "lssmote"])
    parser.add_argument("--oversample_iters", type=int, default=1)
    parser.add_argument("--thresh_type", type=str, default="mean",
                        choices=["mean"] + ["percentile_{}".format(i) for i in range(1,100)])
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

    train_framewise_sensor(args.split_path,
                           args.emb_dir,
                           out_dir,
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
                           thresh_type=args.thresh_type,
                           overwrite=(not args.no_overwrite))
