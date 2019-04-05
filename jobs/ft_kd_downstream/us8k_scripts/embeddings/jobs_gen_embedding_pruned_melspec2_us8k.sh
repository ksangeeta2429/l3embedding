#!/usr/bin/env bash

### Old (no finetuning/KD) ###
### TODO: These need updating in accordance with new script structutres (see below)
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

### New (no FT/KD) ###
sbatch gen_embedding_untuned_pruned_melspec2_us8k.sbatch pruned_audio_0.4974517822265625.h5
sleep 1
sbatch gen_embedding_untuned_pruned_melspec2_us8k.sbatch pruned_audio_0.4986114501953125.h5
sleep 1
sbatch gen_embedding_untuned_pruned_melspec2_us8k.sbatch pruned_audio_0.4989776611328125.h5

### New (with finetuning and KD) ###
sbatch gen_embedding_pruned_melspec2_us8k.sbatch /scratch/dr2915/l3pruning/embedding/fixed/model_best_valid_accuracy_20190117222133.h5 0.0 0.034713585 0.02258205 0.014317851 0.0086109005 0.012467595 0.013169089 0.0132640535
sleep 1
sbatch gen_embedding_pruned_melspec2_us8k.sbatch /scratch/dr2915/l3pruning/embedding/fixed/model_best_valid_accuracy_20190117222227.h5 0.0 0.017678589 0.012369654 0.010109939 0.004976296 0.006389169 0.005634685 0.0067510903
sleep 1
sbatch gen_embedding_pruned_melspec2_us8k.sbatch /scratch/dr2915/l3pruning/embedding/fixed/model_best_valid_accuracy_20190119124017.h5 0.0 0.058792762 0.042968187 0.029552849 0.024852263 0.021678694 0.020085195 0.023608074
sleep 1
sbatch gen_embedding_pruned_melspec2_us8k.sbatch /scratch/dr2915/l3pruning/embedding/fixed/model_best_valid_accuracy_20190119150752.h5 0.08239212 0.034713585 0.02258205 0.016543707 0.0062475526 0.010327266 0.00908012 0.010790716
sleep 1
sbatch gen_embedding_pruned_melspec2_us8k.sbatch /scratch/dr2915/l3pruning/embedding/fixed/model_best_valid_loss_20190121105950.h5 0.0 0.034713585 0.02258205 0.014317851 0.0086109005 0.012467595 0.013169089 0.0132640535

### Newest models (with finetuning and KD) ###
sbatch gen_embedding_pruned_melspec2_us8k.sbatch model_best_valid_accuracy_20190129142215.h5 0.0 0.046718758 0.032320194 0.038394287 0.036401562 0.029490836 0.024827732 0.027749654
sleep 1
sbatch gen_embedding_pruned_melspec2_us8k.sbatch model_best_valid_accuracy_20190129133816.h5 0.0 0.046718758 0.030789277 0.021265747 0.017178306 0.013102217 0.013169089 0.012179388
sleep 1
sbatch gen_embedding_pruned_melspec2_us8k.sbatch model_best_valid_loss_20190128201020.h5 0.0 0.058792762 0.042968187 0.029552849 0.024852263 0.021678694 0.020085195 0.023608074
sleep 1
sbatch gen_embedding_pruned_melspec2_us8k.sbatch model_best_valid_loss_20190128211022.h5 0.0 0.010306876 0.009743502 0.008146217 0.0038607703 0.006389169 0.005634685 0.0067510903
sleep 1
sbatch gen_embedding_pruned_melspec2_us8k.sbatch model_best_valid_accuracy_20190129135949.h5 0.0 0.051247526 0.03440502 0.02266353 0.017178306 0.01508353 0.014082865 0.0132640535
