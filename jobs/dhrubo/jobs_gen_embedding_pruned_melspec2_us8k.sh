#!/usr/bin/env bash

sbatch --export=filename=pruned_model/pruned_audio_0.71586.h5 gen_embedding_pruned_melspec2_us8k.sbatch
sleep 1
sbatch --export=filename=pruned_model/pruned_audio_0.75373.h5 gen_embedding_pruned_melspec2_us8k.sbatch
sleep 1
sbatch --export=filename=pruned_model/pruned_audio_0.77128.h5 gen_embedding_pruned_melspec2_us8k.sbatch
sleep 1
sbatch --export=filename=pruned_model/pruned_audio_0.51833.h5 gen_embedding_pruned_melspec2_us8k.sbatch
sleep 1
sbatch --export=filename=pruned_model/pruned_audio_0.571075.h5 gen_embedding_pruned_melspec2_us8k.sbatch
sleep 1
sbatch --export=filename=pruned_model/pruned_audio_0.685913.h5 gen_embedding_pruned_melspec2_us8k.sbatch
sleep 1
sbatch --export=filename=pruned_model/pruned_audio_0.5031585693359375.h5 gen_embedding_pruned_melspec2_us8k.sbatch
sleep 1
sbatch --export=filename=pruned_model/pruned_audio_0.49981689453125.h5 gen_embedding_pruned_melspec2_us8k.sbatch
sleep 1
sbatch --export=filename=pruned_model/pruned_audio_0.4983367919921875.h5 gen_embedding_pruned_melspec2_us8k.sbatch
