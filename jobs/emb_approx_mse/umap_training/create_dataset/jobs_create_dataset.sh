#!/usr/bin/env bash

sbatch create_training_dataset_umap.sbatch music 2048000
sleep 5
sbatch create_training_dataset_umap.sbatch music 3072000
sleep 5
sbatch create_training_dataset_umap.sbatch music 5120000