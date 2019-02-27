import csv
import logging
import os

import data.usc.features as cls_features

LOGGER = logging.getLogger('cls-data-generation')
LOGGER.setLevel(logging.DEBUG)


NUM_FOLDS = 10

def load_us8k_metadata(path):
    """
    Load UrbanSound8K metadata
    Args:
        path: Path to metadata csv file
              (Type: str)
    Returns:
        metadata: List of metadata dictionaries
                  (Type: list[dict[str, *]])
    """
    metadata = [{} for _ in range(NUM_FOLDS)]
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fname = row['slice_file_name']
            row['start'] = float(row['start'])
            row['end'] = float(row['end'])
            row['salience'] = float(row['salience'])
            fold_num = row['fold'] = int(row['fold'])
            row['classID'] = int(row['classID'])
            metadata[fold_num-1][fname] = row

    return metadata

def generate_us8k_file_data(fname, example_metadata, audio_fold_dir, features,
                            l3embedding_model, **feature_args):
    audio_path = os.path.join(audio_fold_dir, fname)

    basename, _ = os.path.splitext(fname)
    #output_path = os.path.join(output_fold_dir, basename + '.npz')

    #if os.path.exists(output_path):
    #    LOGGER.info('File {} already exists'.format(output_path))
    #    return

    X = cls_features.compute_file_features(audio_path, features, l3embedding_model=l3embedding_model, **feature_args)

    # If we were not able to compute the features, skip this file
    if X is None:
        LOGGER.error('Could not generate data for {}'.format(audio_path))
        return

    class_label = example_metadata['classID']
    y = class_label
    return X, y

    #np.savez_compressed(output_path, X=X, y=y)

    #return output_path, 'success'


