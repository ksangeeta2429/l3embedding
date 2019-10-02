import argparse
import logging
import os.path
from l3embedding.save_approx_embedding import generate_trained_umap_embeddings_driver

def parse_arguments():
    parser = argparse.ArgumentParser(description='Does dimensionality reduction on l3-embeddings and saves it!')

    parser.add_argument('-am',
                        '--approx-mode',
                        dest='approx_mode',
                        action='store',
                        type=str,
                        default='umap',
                        help='Type of embedding approximation method to use: `umap` or `tsne`')

    parser.add_argument('-bs',
                        '--batch-size',
                        dest='batch_size',
                        action='store',
                        type=int,
                        default=1024,
                        help='Number of examples per batch')

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
                        help='Random seed used to set the RNG state')

    parser.add_argument('umap_estimator_path',
                        action='store',
                        type=str,
                        help='UMAP trained model path')

    parser.add_argument('data_dir',
                        action='store',
                        type=str,
                        help='Path to directory where data files are stored')

    parser.add_argument('output_dir',
                        action='store',
                        type=str,
                        help='Path to directory where embeddings will be stored')

    parser.add_argument('reduced_emb_len',
                        action='store',
                        type=int,
                        help='Reduced embedding length')


    return vars(parser.parse_args())


if __name__ == '__main__':
    generate_trained_umap_embeddings_driver(**(parse_arguments()))

