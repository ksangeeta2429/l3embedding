import argparse
import logging
import os.path
from l3embedding.save_approx_embedding import embedding_generator

def parse_arguments():
    parser = argparse.ArgumentParser(description='Does dimensionality reduction on l3-embeddings and saves it!')
                    
    parser.add_argument('-bs',
                        '--batch-size',
                        dest='batch_size',
                        action='store',
                        type=int,
                        default=1024,
                        help='Number of examples per batch')

    parser.add_argument('-am',
                        '--approx-mode',
                        dest='approx_mode',
                        action='store',
                        type=str,
                        default='umap',
                        help='Type of embedding approximation method to use: `umap` or `tsne`')

    parser.add_argument('--neighbors-list',
                        dest='neighbors_list',
                        type=int,
                        default=[],
                        nargs='+',
                        help='Corresponds to list of n_neighbors if approx_mode = `umap` and perplexity if approx_mode = `t-SNE`')

    parser.add_argument('-mdist',
                        '--min-dist-list',
                        dest='min_dist_list',
                        action='store',
                        type=float,
                        default=[0.3],
                        nargs='*',
                        help='UMAP: Minimum distance between clusters. \
                              Sensible values are in the range 0.001 to 0.5, with 0.1 being a reasonable default.')

    parser.add_argument('-metric',
                        '--metric-list',
                        dest='metric_list',
                        action='store',
                        type=str,
                        default=[],
                        nargs='*',
                        help='Optimization metric for UMAP/t-SNE.')

    parser.add_argument('-iter',
                        '--tsne-iter-list',
                        dest='tsne_iter_list',
                        action='store',
                        type=int,
                        default=[],
                        nargs='*',
                        help='Number of iterations for training t-SNE.')

    parser.add_argument('-r',
                        '--random-state',
                        dest='random_state',
                        action='store',
                        type=int,
                        default=20171021,
                        help='Random seed used to set the RNG state')

    parser.add_argument('-trans-dir',
                        '--transform-data-dir',
                        dest='transform_data_dir',
                        action='store',
                        type=str,
                        help='Path to directory where data to be transformed is stored')

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
    embedding_generator(**(parse_arguments()))

