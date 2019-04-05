import argparse
import logging
import os
import keras
import json
from keras.optimizers import Adam
from kapre.time_frequency import Melspectrogram
from l3embedding.model import load_embedding
from data.usc.dcase2013 import generate_dcase2013_folds, generate_dcase2013_fold_data
from data.usc.esc50 import generate_esc50_folds, generate_esc50_fold_data
from data.usc.us8k import generate_us8k_folds, generate_us8k_fold_data
from log import init_console_logger

LOGGER = logging.getLogger('cls-data-generation')
LOGGER.setLevel(logging.DEBUG)


def parse_arguments():
    """
    Parse arguments from the command line


    Returns:
        args:  Argument dictionary
               (Type: dict[str, *])
    """
    parser = argparse.ArgumentParser(description='Train an urban sound classification model')

    parser.add_argument('-r',
                        '--random-state',
                        dest='random_state',
                        action='store',
                        type=int,
                        default=20171021,
                        help='Random seed used to set the RNG state')

    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action='store_true',
                        default=False,
                        help='If True, print detailed messages')

    parser.add_argument('-f',
                        '--features',
                        dest='features',
                        action='store',
                        type=str,
                        default='l3',
                        help='Type of features to be used in training')

    parser.add_argument('-lmp',
                        '--l3embedding-model-path',
                        dest='l3embedding_model_path',
                        action='store',
                        type=str,
                        help='Path to L3 embedding model weights file')


    parser.add_argument('-lpt',
                        '--l3embedding-pooling-type',
                        dest='l3embedding_pooling_type',
                        action='store',
                        type=str,
                        default='original',
                        help='Type of pooling used to downsample last conv layer of L3 embedding model')

    parser.add_argument('-lmt',
                        '--l3embedding-model-type',
                        dest='l3embedding_model_type',
                        action='store',
                        type=str,
                        default='cnn_L3_melspec2',
                        help='L3 embedding model type')

    parser.add_argument('-fcl',
                        '--from-conv-layer',
                        dest='from_conv_layer',
                        action='store',
                        type=int,
                        default=8,
                        help='Conv. layer to derive embedding from (1-8)')

    parser.add_argument('-hs',
                        '--hop-size',
                        dest='hop_size',
                        action='store',
                        type=float,
                        default=0.1,
                        help='Hop size in seconds')

    parser.add_argument('-srate',
                        '--samp-rate',
                        dest='samp_rate',
                        action='store',
                        type=int,
                        default=48000,
                        help='Sampling rate')

    parser.add_argument('-nmels',
                        '--num-mels',
                        dest='n_mels',
                        action='store',
                        type=int,
                        default=256,
                        help='Number of mel filters')

    parser.add_argument('-lhop',
                        '--hop-length',
                        dest='n_hop',
                        action='store',
                        type=int,
                        default=242,
                        help='Melspec hop length in samples')

    parser.add_argument('-ndft',
                        '--num-dft',
                        dest='n_dft',
                        action='store',
                        type=int,
                        default=2048,
                        help='DFT size')

    parser.add_argument('-nrs',
                        '--num-random-samples',
                        dest='num_random_samples',
                        action='store',
                        type=int,
                        help='Number of random samples for randomized sampling methods')

    parser.add_argument('-g',
                        '--gpus',
                        dest='gpus',
                        type=int,
                        default=0,
                        help='Number of gpus used for running the embedding model.')


    parser.add_argument('-filters',
                        '--num_filters',
                        dest='num_filters',
                        action='store',
                        default=None,
                        type=int,
                        nargs='+',
                        help='Set the new number of filters for filterwise pruning')

    parser.add_argument('-layers',
                        '--include_layers',
                        dest='include_layers',
                        action='store',
                        default=None,
                        type=int,
                        nargs='+',
                        help='Select the layers to be included in the new audio model')

    parser.add_argument('--fold',
                        dest='fold',
                        type=int,
                        help='Fold number to generate. If unused, generate all folds')

    parser.add_argument('-th',
                        '--thresholds',
                        dest='thresholds',
                        action='store',
                        default=None,
                        type=float,
                        nargs='+',
                        help='Set the sparsity list for layerwise pruning')

    parser.add_argument('-ump',
                        '--us8k-metadata-path',
                        dest='us8k_metadata_path',
                        type=str,
                        action='store',
                        help='Path to UrbanSound8K metadata file')

    parser.add_argument('dataset_name',
                        action='store',
                        type=str,
                        choices=['us8k', 'esc50', 'dcase2013'],
                        help='Name of dataset')

    parser.add_argument('data_dir',
                        action='store',
                        type=str,
                        help='Path to directory where training set files are stored')

    parser.add_argument('output_dir',
                        action='store',
                        type=str,
                        help='Path to directory where output data files will be stored')

    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_arguments()

    init_console_logger(LOGGER, verbose=args['verbose'])
    LOGGER.debug('Initialized logging.')

    # Unpack CL args
    model_type = args['l3embedding_model_type']
    pooling_type = args['l3embedding_pooling_type']
    metadata_path = args['us8k_metadata_path']
    data_dir = args['data_dir']
    features = args['features']
    hop_size = args['hop_size']
    random_state = args['random_state']
    num_random_samples = args['num_random_samples']
    model_path = args['l3embedding_model_path']
    num_gpus = args['gpus']
    output_dir = args['output_dir']
    dataset_name = args['dataset_name']
    fold_num = args['fold']
    from_conv_layer = args['from_conv_layer']
    thresholds = args['thresholds']
    layers = args['include_layers']
    filters = args['num_filters']
    samp_rate = args['samp_rate']
    n_mels = args['n_mels']
    n_hop = args['n_hop']
    n_dft = args['n_dft']

    LOGGER.info('Configuration: {}'.format(str(args)))

    is_l3_feature = features == 'l3'
    is_l3_comp = features == 'l3comp'
    if (is_l3_feature or is_l3_comp) and not model_path:
        raise ValueError('Must provide model path is L3 embedding features are used')


    if is_l3_comp:
        if 'fixed' in model_path:
        # Only get model name and enclosing directory
            short_model_path = model_path[model_path.rindex('fixed'):]
        else:
            short_model_path = model_path

        # If using an L3 model, make model arch. type and pooling type to path
        if from_conv_layer==8:
            embedding_desc_str = short_model_path.replace('.h5', '')
        else:
            embedding_desc_str = os.path.join(os.path.dirname(short_model_path), 'from_convlayer_' + str(from_conv_layer),\
                                              os.path.basename(short_model_path).replace('.h5', ''))

        # Get output dir
        dataset_output_dir = os.path.join(output_dir, 'features', dataset_name,
                                          features, pooling_type, embedding_desc_str)

        if 'masked' in model_type:
            assert thresholds is not None

        if 'reduced' in model_type:
            assert layers is not None or filters is not None

        # Load L3 embedding model if using L3 features
        LOGGER.info('Loading embedding model...')
        l3embedding_model = load_embedding(model_path, model_type, 'audio', pooling_type,
                                           tgt_num_gpus=num_gpus, thresholds=thresholds, \
                                           include_layers=layers, num_filters=filters, from_convlayer=from_conv_layer,
                                           n_mels=n_mels, n_hop=n_hop, n_dft=n_dft, asr=samp_rate)

    elif is_l3_feature:
        # Get output dir
        model_desc_start_idx = model_path.rindex('embedding')+10
        model_desc_end_idx = os.path.dirname(model_path).rindex('/')
        embedding_desc_str = model_path[model_desc_start_idx:model_desc_end_idx]
        # If using an L3 model, make model arch. type and pooling type to path
        dataset_output_dir = os.path.join(output_dir, 'features', dataset_name,
                                          features, pooling_type, embedding_desc_str)
        print('Output directory:',dataset_output_dir)
        # Load L3 embedding model if using L3 features
        LOGGER.info('Loading embedding model...')
        #model_type = embedding_desc_str.split('/')[-1]
        """l3embedding_model = load_embedding(model_path,
                                           model_type,
                                           'audio', pooling_type,
                                           tgt_num_gpus=num_gpus,
                                           n_mels=n_mels, n_hop=n_hop, n_dft=n_dft, asr=samp_rate)"""
        model = keras.models.load_model(model_path, custom_objects={'Melspectrogram': Melspectrogram})
        POOLINGS = {
            'cnn_L3_melspec2': {
                '16k_64_50': (8, 6),
                '48K_64_968_2048': (8, 6),
                '48K_64_242_2048': (8, 24),
                '48K_64_1936_2048': (8, 3),
                '48K_128_968_2048': (16, 6),
                '48K_128_1936_2048': (16, 3),
                '48K_256_484_2048': (32, 12),
                '48K_256_968_2048': (32, 6),
                '48K_256_1936_2048': (32, 3),
                '16K_64_968_1024': (8, 2),
                '16K_64_484_1024': (8, 4),
            }
        }        

        pool_size = POOLINGS[model_type][pooling_type]

        y_a = keras.layers.MaxPooling2D(pool_size=pool_size, padding='same')(model.output)
        y_a = keras.layers.Flatten()(y_a)
        l3embedding_model = keras.models.Model(inputs=model.input, outputs=y_a)
       
        #print("------------------------------------------------------------------")
        #print(model.summary())
        #print("------------------------------------------------------------------")
        #print(l3embedding_model.summary())

    else:
        # Get output dir
        dataset_output_dir = os.path.join(output_dir, 'features', dataset_name, features)
        l3embedding_model = None

    # Make sure output directory exists
    if not os.path.isdir(dataset_output_dir):
        os.makedirs(dataset_output_dir)

    args['features_dir'] = dataset_output_dir
    # Write configurations to a file for reproducibility/posterity
    config_path = os.path.join(dataset_output_dir, 'config_{}.json'.format(fold_num))
    with open(config_path, 'w') as f:
        json.dump(args, f)
    LOGGER.info('Saved configuration to {}'.format(config_path))

    # Resetting features to 'l3'
    if is_l3_comp:
        features='l3'

    if dataset_name == 'us8k':
        if not metadata_path:
            raise ValueError('Must provide metadata file for UrbanSound8k')

        if fold_num is not None:
            # Generate a single fold if a fold was specified
            generate_us8k_fold_data(metadata_path, data_dir, fold_num-1, dataset_output_dir,
                                    l3embedding_model=l3embedding_model,
                                    features=features, random_state=random_state,
                                    hop_size=hop_size, num_random_samples=num_random_samples, mel_hop_length=n_hop, n_mels=n_mels, n_fft=n_dft, sr=samp_rate)

        else:
            # Otherwise, generate all the folds
            generate_us8k_folds(metadata_path, data_dir, dataset_output_dir,
                                l3embedding_model=l3embedding_model,
                                features=features, random_state=random_state,
                                hop_size=hop_size, num_random_samples=num_random_samples, mel_hop_length=n_hop, n_mels=n_mels, n_fft=n_dft, sr=samp_rate)

    elif dataset_name == 'esc50':
        if fold_num is not None:
            generate_esc50_fold_data(data_dir, fold_num-1, dataset_output_dir,
                                     l3embedding_model=l3embedding_model, features=features, 
				     random_state=random_state, hop_size=hop_size, num_random_samples=num_random_samples, 
				     mel_hop_length=n_hop, n_mels=n_mels, n_fft=n_dft, samp_rate=samp_rate)
        else:
            generate_esc50_folds(data_dir, dataset_output_dir,
                                 l3embedding_model=l3embedding_model, features=features, random_state=random_state, 
                                 num_random_samples=num_random_samples, mel_hop_length=n_hop, n_mels=n_mels, n_fft=n_dft, samp_rate=samp_rate)

    elif dataset_name == 'dcase2013':
        if fold_num is not None:
            generate_dcase2013_fold_data(data_dir, fold_num-1, dataset_output_dir,
                                         l3embedding_model=l3embedding_model,
                                         features=features, random_state=random_state,
                                         hop_size=hop_size, num_random_samples=num_random_samples, samp_rate=samp_rate)
        else:
            generate_dcase2013_folds(data_dir, dataset_output_dir,
                                     l3embedding_model=l3embedding_model,
                                     features=features, random_state=random_state,
                                     hop_size=hop_size, num_random_samples=num_random_samples, samp_rate=samp_rate)

    else:
        LOGGER.error('Invalid dataset name: {}'.format(dataset_name))

    LOGGER.info('Done!')
