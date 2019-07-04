import argparse
import logging
import os.path
from l3embedding.distillation import train


def parse_arguments():
    """
    Parse arguments from the command line
    Returns:
        args:  Argument dictionary
               (Type: dict[str, *])
    """
    parser = argparse.ArgumentParser(description='Train an L3-like audio-visual correspondence model')

    parser.add_argument('-lt',
                        '--loss-type',
                        dest='loss_type',
                        action='store',
                        type=str,
                        default='entropy',
                        help='`entropy` for crossentropy on AVC, `mse` for embedding approximation')

    parser.add_argument('-temp',
                        '--temperature',
                        dest='temp',
                        action='store',
                        type=int,
                        default=4,
                        help='Temperature for smoothing logits. Value of 1 corresponds to softmax.')

    parser.add_argument('-lambda',
                        '--lambda-constant',
                        dest='lambda_constant',
                        action='store',
                        type=float,
                        default=0.9,
                        help='Weight factor for softened crossentropy')

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
                        default='cnn_L3_orig',
                        help='Name of model type to train')

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

    parser.add_argument('student_weight_path',
                        action='store',
                        type=str,
                        help='Path to model weights directory')

    parser.add_argument('teacher_weight_path',
                        action='store',
                        type=str,
                        help='Path to model weights directory')

    parser.add_argument('output_dir',
                        action='store',
                        type=str,
                        help='Path to directory where output files will be stored')


    return vars(parser.parse_args())


if __name__ == '__main__':
    train(**(parse_arguments()))
