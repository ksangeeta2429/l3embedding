#!/usr/bin/env bash

# /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.51833/

sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.51833/ 1
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.51833/ 2
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.51833/ 3
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.51833/ 4
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.51833/ 5
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.51833/ 6
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.51833/ 7
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.51833/ 8
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.51833/ 9
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.51833/ 10