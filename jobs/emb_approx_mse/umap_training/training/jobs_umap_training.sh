#!/usr/bin/env bash

#sbatch umap_training.sbatch music 1000000 256 5 0.3 correlation
#sbatch umap_training.sbatch music 1500000 256 5 0.3 correlation
#
#sbatch big_batch_umap_training.sbatch music 5000000 256 5 0.3 correlation
#sbatch big_batch_umap_training.sbatch music 10000000 256 5 0.3 correlation

# Euclidean -- music dataset
# sbatch umap_training.sbatch music 2048000 256 5 0.3 -- Error: numpy.linalg.linalg.LinAlgError: 173-th leading minor not positive definite
#sbatch umap_training.sbatch music 5120000 256 500 0.3
#sbatch umap_training.sbatch music 5120000 256 300 0.3
sbatch umap_training.sbatch music 2048000 256 2000 0.3
sleep 3
sbatch umap_training.sbatch music 2048000 256 5000 0.3
#sbatch umap_training.sbatch music 2048000 256 20 0.3
#sbatch umap_training.sbatch music 2048000 256 30 0.3
#sbatch umap_training.sbatch music 2048000 256 40 0.3
#sbatch umap_training.sbatch music 2048000 256 50 0.3

# Euclidean -- sonyc dataset
#sbatch umap_training.sbatch sonyc 2048000 256 10 0.3
#sbatch umap_training.sbatch sonyc 2048000 256 20 0.3
#sbatch umap_training.sbatch sonyc 2048000 256 30 0.3
#sbatch umap_training.sbatch sonyc 2048000 256 40 0.3
#sbatch umap_training.sbatch sonyc 2048000 256 50 0.3
