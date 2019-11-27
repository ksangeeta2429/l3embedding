import argparse
import logging
import os.path
from l3embedding.save_orig_l3_embedding import generate_output_driver

def parse_arguments():
    parser = argparse.ArgumentParser(description='Saves original l3-embeddings!')
                    
    parser.add_argument('-r',
                        '--random-state',
                        dest='random_state',
                        action='store',
                        type=int,
                        default=20171021,
                        help='Random seed used to set the RNG state')

    parser.add_argument('-pt',
                        '--partition-num',
                        dest='partition_to_run',
                        action='store',
                        type=int,
                        default=None,
                        help='Chunk index to run')

    parser.add_argument('-si',
                        '--start-idx',
                        dest='start_idx',
                        action='store',
                        type=int,
                        default=None,
                        help='Starting index of the file to run')

    parser.add_argument('-out_type',
                        '--out-type',
                        action='store',
                        type=str,
                        help='Choose from "l3_embedding" and "logits"')

    parser.add_argument('data_dir',
                        action='store',
                        type=str,
                        help='Path to directory where data files are stored')

    parser.add_argument('output_dir',
                        action='store',
                        type=str,
                        help='Path to directory where reduced embeddings will be stored')

    return vars(parser.parse_args())


if __name__ == '__main__':
    generate_output_driver(**(parse_arguments()))

