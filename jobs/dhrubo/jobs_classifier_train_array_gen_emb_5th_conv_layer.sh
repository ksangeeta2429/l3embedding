#!/usr/bin/env bash

sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/cnn_l3_melspec2_recent/from_convlayer_5/model_best_valid_accuracy/ 1
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/cnn_l3_melspec2_recent/from_convlayer_5/model_best_valid_accuracy/ 2
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/cnn_l3_melspec2_recent/from_convlayer_5/model_best_valid_accuracy/ 3
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/cnn_l3_melspec2_recent/from_convlayer_5/model_best_valid_accuracy/ 4
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/cnn_l3_melspec2_recent/from_convlayer_5/model_best_valid_accuracy/ 5
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/cnn_l3_melspec2_recent/from_convlayer_5/model_best_valid_accuracy/ 6
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/cnn_l3_melspec2_recent/from_convlayer_5/model_best_valid_accuracy/ 7
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/cnn_l3_melspec2_recent/from_convlayer_5/model_best_valid_accuracy/ 8
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/cnn_l3_melspec2_recent/from_convlayer_5/model_best_valid_accuracy/ 9
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/models/cnn_l3_melspec2_recent/from_convlayer_5/model_best_valid_accuracy/ 10
