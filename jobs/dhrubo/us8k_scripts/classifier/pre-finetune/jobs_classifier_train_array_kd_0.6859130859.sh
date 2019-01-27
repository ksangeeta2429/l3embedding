#!/usr/bin/env bash

# /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/pruning_kd/latest_model

sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/pruning_kd/latest_model 1
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/pruning_kd/latest_model 2
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/pruning_kd/latest_model 3
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/pruning_kd/latest_model 4
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/pruning_kd/latest_model 5
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/pruning_kd/latest_model 6
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/pruning_kd/latest_model 7
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/pruning_kd/latest_model 8
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/pruning_kd/latest_model 9
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/pruning_kd/latest_model 10