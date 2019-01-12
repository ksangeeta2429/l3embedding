#!/usr/bin/env bash

# /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.75373/

sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.75373/ 1
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.75373/ 2
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.75373/ 3
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.75373/ 4
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.75373/ 5
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.75373/ 6
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.75373/ 7
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.75373/ 8
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.75373/ 9
sleep 1
sbatch classifier-train-array-us8k.sbatch /scratch/dr2915/embeddings/features/us8k/l3comp/short/pruned_model/pruned_audio_0.75373/ 10