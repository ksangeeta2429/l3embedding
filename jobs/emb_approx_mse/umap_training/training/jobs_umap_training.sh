#!/usr/bin/env bash

sbatch umap_training.sbatch music 1000000 256 5 0.3 correlation
sbatch umap_training.sbatch music 1500000 256 5 0.3 correlation

sbatch big_batch_umap_training.sbatch music 5000000 256 5 0.3 correlation
sbatch big_batch_umap_training.sbatch music 10000000 256 5 0.3 correlation