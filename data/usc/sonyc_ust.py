import csv
import logging
import os
import glob
import random
import numpy as np
import pandas as pd

import data.usc.features as cls_features
from log import LogTimer

LOGGER = logging.getLogger('cls-data-generation')
LOGGER.setLevel(logging.DEBUG)

def generate_sonyc_ust_data(annotation_path, dataset_dir, output_dir, l3embedding_model,
                            hop_size=0.1, features='l3', timestamps = False, **feature_args):
    """
    Extract embeddings for files annotated in the SONYC annotation file and save them to disk.
    Parameters
    ----------
    annotation_path: str
        Path to SONYC_UST annotation csv file
    dataset_dir: str
        Path to SONYC_UST dataset
    output_dir : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    l3embedding_model : keras.models.Model
        Model Object
    hop_size : float
        Hop size in seconds

    Returns
    -------
    """

    print("* Loading annotations.")
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')

    # out_dir = os.path.join(output_dir, 'l3-{}-{}-{}'.format(input_repr, content_type, embedding_size))
    os.makedirs(output_dir, exist_ok=True)

    df = annotation_data[['split', 'audio_filename']].drop_duplicates()
    row_iter = df.iterrows()

    LOGGER.info('* Extracting embeddings.')

    for _, row in row_iter:
        filename = row['audio_filename']
        split_str = row['split']
        audio_path = os.path.join(dataset_dir, split_str, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.npz')

        if not os.path.exists(audio_path):
            LOGGER.info('Audio file {} doesn''t exist'.format(audio_path))
            continue

        if os.path.exists(output_path):
            LOGGER.info('Output file {} already exists'.format(output_path))
            return

        X = cls_features.compute_file_features(audio_path, features, l3embedding_model=l3embedding_model,
                                               hop_size=hop_size, **feature_args)

        # If we were not able to compute the features, skip this file
        if X is None:
            LOGGER.error('Could not generate data for {}'.format(audio_path))
            return

        if timestamps:
            # Save timestamps as well, if necessary
            ts = np.arange(X.shape[0]) * hop_size
            np.savez(output_path, embedding=X, timestamps=ts)
        else:
            np.savez(output_path, embedding=X)