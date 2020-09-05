import argparse
import logging
import os.path
from l3embedding.train_approx_embedding_mse import train

def parse_arguments():
    """
    Parse arguments from the command line
    Returns:
        args:  Argument dictionary
               (Type: dict[str, *])
    """
    parser = argparse.ArgumentParser(description='Train an L3-like audio-visual correspondence model')

    parser.add_argument('-e',
                        '--num-epochs',
                        dest='num_epochs',
                        action='store',
                        type=int,
                        default=150,
                        help='Maximum number of training epochs')

    parser.add_argument('-tes',
                        '--train-epoch-size',
                        dest='train_epoch_size',
                        action='store',
                        type=int,
                        default=512,
                        help='Number of training batches per epoch')

    parser.add_argument('-ves',
                        '--validation-epoch-size',
                        dest='validation_epoch_size',
                        action='store',
                        type=int,
                        default=1024,
                        help='Number of validation batches per epoch')

    parser.add_argument('-tbs',
                        '--train-batch-size',
                        dest='train_batch_size',
                        action='store',
                        type=int,
                        default=64,
                        help='Number of examples per training batch')

    parser.add_argument('-vbs',
                        '--validation-batch-size',
                        dest='validation_batch_size',
                        action='store',
                        type=int,
                        default=64,
                        help='Number of examples per  batch')

    parser.add_argument('-lr',
                        '--learning-rate',
                        dest='learning_rate',
                        action='store',
                        type=float,
                        default=1e-4,
                        help='Optimization learning rate')

    parser.add_argument('-mt',
                        '--model-type',
                        dest='model_type',
                        action='store',
                        type=str,
                        default='cnn_L3_melspec2',
                        help='Name of model type to train')

    parser.add_argument('-melSpec',
                        '--melSpec',
                        dest='melSpec',
                        action='store_true',
                        default=False,
                        help='Set to True is Melspec is not included in the model')

    parser.add_argument('-student',
                        '--student-weight-path',
                        dest='student_weight_path',
                        action='store',
                        type=str,
                        help='Path to the student weight file')

    parser.add_argument('-am',
                        '--approx-mode',
                        dest='approx_mode',
                        action='store',
                        type=str,
                        default='umap',
                        help='Type of reduction: `umap` or `tsne`')

    # parser.add_argument('-ats',
    #                     '--approx-train-size',
    #                     dest='approx_train_size',
    #                     action='store',
    #                     type=int,
    #                     default=2000000,
    #                     help='Number of examples used to train the approximated method')
    
    # parser.add_argument('-neighbors',
    #                     '--neighbors',
    #                     dest='neighbors',
    #                     type=int,
    #                     default=20,
    #                     help='Corresponds to n_neighbors if approx_mode = `umap` and perplexity if approx_mode = `tsne`. Possible values = {5, 10, 20, 30}')

    # parser.add_argument('-metric',
    #                     '--metric',
    #                     dest='metric',
    #                     action='store',
    #                     type=str,
    #                     default='correlation',
    #                     help='Type of metric to optimize with umap/tsne: {correlation}')

    # parser.add_argument('-mdist',
    #                     '--min-dist',
    #                     dest='min_dist',
    #                     action='store',
    #                     type=float,
    #                     default=0.3,
    #                     help='UMAP: Minimum distance between clusters. Sensible values are in the range 0.001 to 0.5, with 0.1 being a reasonable default.')

    parser.add_argument('-srate',
                        '--samp-rate',
                        dest='samp_rate',
                        action='store',
                        type=int,
                        default=8000,
                        help='Sampling rate')

    parser.add_argument('-nmels',
                        '--num-mels',
                        dest='n_mels',
                        action='store',
                        type=int,
                        default=64,
                        help='Number of mel filters')

    parser.add_argument('-lhop',
                        '--hop-length',
                        dest='n_hop',
                        action='store',
                        type=int,
                        default=160,
                        help='Hop length in samples')

    parser.add_argument('-ndft',
                        '--num-dft',
                        dest='n_dft',
                        action='store',
                        type=int,
                        default=1024,
                        help='DFT size')

    parser.add_argument('-fmax',
                        '--freq-max',
                        dest='fmax',
                        action='store',
                        type=int,
                        default=None,
                        help='Max. freq in DFT')

    parser.add_argument('-half',
                        '--halved-filters',
                        dest='halved_convs',
                        action='store_true',
                        default=True,
                        help='Use half the number of conv. filters as in the original audio model?')

    parser.add_argument('-ci',
                        '--checkpoint-interval',
                        dest='checkpoint_interval',
                        action='store',
                        type=int,
                        default=10,
                        help='The number of epochs between model checkpoints')

    parser.add_argument('-r',
                        '--random-state',
                        dest='random_state',
                        action='store',
                        type=int,
                        default=20171021,
                        help='Random seed used to set the RNG state')

    parser.add_argument('--gpus',
                        dest='gpus',
                        type=int,
                        default=1,
                        help='Number of gpus used for data parallelism.')

    parser.add_argument('-gsid',
                        '--gsheet-id',
                        dest='gsheet_id',
                        type=str,
                        help='Google Spreadsheet ID for centralized logging of experiments')

    parser.add_argument('-gdan',
                        '--google-dev-app-name',
                        dest='google_dev_app_name',
                        type=str,
                        help='Google Developer Application Name for using API')

    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action='store_true',
                        default=False,
                        help='If True, print detailed messages')

    parser.add_argument('-cmd',
                        '--continue-model-dir',
                        dest='continue_model_dir',
                        action='store',
                        type=str,
                        help='Path to directory containing a model with which to resume training')

    parser.add_argument('-lp',
                        '--log-path',
                        dest='log_path',
                        action='store',
                        default=None,
                        help='Path to log file generated by this script. ' \
                             'By default, the path is "./l3embedding.log".')

    parser.add_argument('-nl',
                        '--no-logging',
                        dest='disable_logging',
                        action='store_true',
                        default=False,
                        help='Disables logging if flag enabled')

    parser.add_argument('train_data_dir',
                        action='store',
                        type=str,
                        help='Path to directory where training set files are stored')

    parser.add_argument('validation_data_dir',
                        action='store',
                        type=str,
                        help='Path to directory where validation set files are stored')

    parser.add_argument('emb_train_dir',
                        action='store',
                        type=str,
                        help='Path to directory where reduced embeddings are stored for training data')

    parser.add_argument('emb_valid_dir',
                        action='store',
                        type=str,
                        help='Path to directory where reduced embeddings are stored for validation data')

    parser.add_argument('output_dir',
                        action='store',
                        type=str,
                        help='Path to directory where output files will be stored')


    return vars(parser.parse_args())


if __name__ == '__main__':
    train(**(parse_arguments()))
