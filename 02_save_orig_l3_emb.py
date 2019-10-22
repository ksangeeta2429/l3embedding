import argparse
import logging
import os.path
from l3embedding.save_orig_l3_embedding import embedding_generator

def parse_arguments():
    parser = argparse.ArgumentParser(description='Saves original l3-embeddings!')
                    
    parser.add_argument('-r',
                        '--random-state',
                        dest='random_state',
                        action='store',
                        type=int,
                        default=20171021,
                        help='Random seed used to set the RNG state')

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
    embedding_generator(**(parse_arguments()))

