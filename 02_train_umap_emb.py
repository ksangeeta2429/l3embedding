import argparse
import logging
import os.path
from l3embedding.save_approx_embedding import train_umap_embedding

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains UMAP dimensionality reduction on l3-embeddings and saves it!')
                    
    parser.add_argument('-bs',
                        '--batch-size',
                        dest='batch_size',
                        action='store',
                        type=int,
                        default=1024,
                        help='Number of examples per batch')

    parser.add_argument('--neighbors',
                        dest='neighbors',
                        type=int,
                        default=5,
                        help='n_neighbors for UMAP')

    parser.add_argument('-mdist',
                        '--min-dist',
                        dest='min_dist',
                        type=float,
                        default=0.3,
                        help='UMAP: Minimum distance between clusters. \
                              Sensible values are in the range 0.001 to 0.5, with 0.1 being a reasonable default.')

    parser.add_argument('-metric',
                        '--metric',
                        dest='metric',
                        type=str,
                        default='euclidean',
                        help='Optimization metric for UMAP')

    parser.add_argument('-r',
                        '--random-state',
                        dest='random_state',
                        action='store',
                        type=int,
                        default=20171021,
                        help='Random seed used to set the RNG state')

    parser.add_argument('data_dir',
                        action='store',
                        type=str,
                        help='Path to directory where data files are stored')

    parser.add_argument('output_dir',
                        action='store',
                        type=str,
                        help='Path to directory where reduced embeddings will be stored')

    parser.add_argument('reduced_emb_len',
                        action='store',
                        type=int,
                        help='Reduced embedding length')


    return vars(parser.parse_args())


if __name__ == '__main__':
    train_umap_embedding(**(parse_arguments()))

