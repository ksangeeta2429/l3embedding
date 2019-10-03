import argparse
import logging
import os.path
from sonyc.utils import downsample_sonyc_points

def parse_arguments():
    parser = argparse.ArgumentParser(description='Creates randomized training dataset for UMAP')

    parser.add_argument('-r',
                        '--random-state',
                        dest='random_state',
                        action='store',
                        type=int,
                        default=20171021,
                        help='Random seed used to set the RNG state')

    parser.add_argument('csv_path',
                        action='store',
                        type=str,
                        help='Path to directory where data files are stored')

    parser.add_argument('output_dir',
                        action='store',
                        type=str,
                        help='Path to directory where sampled training will be stored')

    parser.add_argument('sample_size',
                        action='store',
                        type=int,
                        help='Size of training set')

    return vars(parser.parse_args())

if __name__ == '__main__':
    downsample_sonyc_points(**(parse_arguments()))