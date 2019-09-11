import argparse
import os
import numpy as np
import yaml
import pandas as pd
from collections import OrderedDict
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam
import keras


def construct_relabel_model(num_labels, l2_reg=1e-5):
    # Input layer
    inp = Input(shape=(num_labels,), dtype='float32')

    # Output layer
    y = Dense(num_labels, activation='sigmoid',
              kernel_regularizer=regularizers.l2(l2_reg))(inp)

    m = Model(inputs=inp, outputs=y)
    print(m.summary())
    return m


def predict_train_labels(model, annotation_path, taxonomy_path):

    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.load(f, Loader=yaml.Loader)

    # Limit to validate (and test) sets
    train_annotation_data = annotation_data[annotation_data['split'] == 'train']

    target_labels = ["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                     for coarse_id, fine_dict in taxonomy['fine'].items()
                     for fine_id, fine_label in fine_dict.items()]

    annotators_label_list = []

    file_list = train_annotation_data.sort_values('audio_filename')['audio_filename'].unique().tolist()

    for filename in file_list:
        file_df = train_annotation_data[train_annotation_data['audio_filename'] == filename]
        annotators_label = []

        for label in target_labels:
            count = 0

            for _, row in file_df.iterrows():
                if int(row['annotator_id']) > 0:
                    count += float(row[label + '_presence'])

            annotators_label.append(count)

        annotators_label_list.append(annotators_label)

    X = np.array(annotators_label_list)
    pred_list = model.predict(X)

    relabeled_df = []
    for filename, pred in zip(file_list, pred_list):
        file_ex = train_annotation_data[train_annotation_data['audio_filename'] == filename].iloc[0]

        row = OrderedDict()

        row['split'] = 'train'
        row['sensor_id'] = file_ex['sensor_id']
        row['audio_filename'] = filename
        row['annotator_id'] = 0

        for label, y in zip(target_labels, pred):
            row[label + '_presence'] = y

        for label in target_labels:
            row[label + '_proximity'] = -1


        for coarse_num, coarse_name in taxonomy['coarse'].items():
            coarse_label = "{}_{}".format(coarse_num, coarse_name)

            max_prob = 0.0
            for fine_num, fine_name in taxonomy['fine'][coarse_num].items():
                label = "{}-{}_{}".format(coarse_num, fine_num, fine_name)
                max_prob = max(max_prob, row[label + '_presence'])

            row[coarse_label + '_presence'] = max_prob

        relabeled_df.append(row)

    relabeled_df = pd.DataFrame(relabeled_df)

    validated_data = annotation_data[(annotation_data['split'] == 'validate') | (annotation_data['split'] == 'test')]
    validated_data = validated_data[validated_data['annotator_id'] == 0]

    relabeled_df = pd.concat([relabeled_df, validated_data])

    return relabeled_df


def load_model_data(annotation_path, taxonomy_path):
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.load(f, Loader=yaml.Loader)

    # Limit to validate (and test) sets
    annotation_data = annotation_data[(annotation_data['split'] == 'validate') | (annotation_data['split'] == 'test')]

    target_labels = ["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                     for coarse_id, fine_dict in taxonomy['fine'].items()
                     for fine_id, fine_label in fine_dict.items()]

    annotators_label_list = []
    verified_label_list = []

    file_list = annotation_data.sort_values('audio_filename')['audio_filename'].unique().tolist()

    for filename in file_list:
        file_df = annotation_data[annotation_data['audio_filename'] == filename]
        annotators_label = []
        verified_label = []

        for label in target_labels:
            count = 0

            for _, row in file_df.iterrows():
                if int(row['annotator_id']) == 0:
                    verified_label.append(float(row[label + '_presence']))
                elif int(row['annotator_id']) > 0:
                    count += float(row[label + '_presence'])

                # Ignore SONYC verifiers

            annotators_label.append(count)

        annotators_label_list.append(annotators_label)
        verified_label_list.append(verified_label)

    return np.array(annotators_label_list), np.array(verified_label_list)


def train_logistic_relabler(annotation_path, taxonomy_path, output_dir, num_epochs=50,
                            lr=1e-5, batch_size=64, l2_reg=1e-5, valid_split=0.1):

    X_train, y_train = load_model_data(annotation_path, taxonomy_path)

    num_labels = y_train.shape[-1]
    model = construct_relabel_model(num_labels, l2_reg=l2_reg)


    opt = Adam(lr)
    model.compile(opt, loss='binary_crossentropy', metrics=['accuracy'])

    cb = []
    # checkpoint
    model_weight_file = os.path.join(output_dir, 'relabel_model_best.h5')
    model.save_weights(model_weight_file)

    cb.append(keras.callbacks.ModelCheckpoint(model_weight_file,
                                              save_weights_only=True,
                                              save_best_only=True,
                                              monitor='val_loss'))
    # monitor losses
    history_csv_file = os.path.join(output_dir, 'history.csv')
    cb.append(keras.callbacks.CSVLogger(history_csv_file, append=True,
                                        separator=','))

    model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
              validation_split=valid_split, callbacks=cb)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_path')
    parser.add_argument('taxonomy_path')
    parser.add_argument('output_dir')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--l2_reg', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--valid_split', type=float, default=0.1)

    args = parser.parse_args()

    model = train_logistic_relabler(args.annotation_path,
                                    args.taxonomy_path,
                                    args.output_dir,
                                    num_epochs=args.num_epochs,
                                    lr=args.lr,
                                    batch_size=args.batch_size,
                                    l2_reg=args.l2_reg,
                                    valid_split=args.valid_split)

    relabled_df = predict_train_labels(model,
                                       args.annotation_path,
                                       args.taxonomy_path)


    output_path = os.path.join(args.output_dir, 'relabeled_annotations.csv')
    relabled_df.to_csv(output_path, index=False)



