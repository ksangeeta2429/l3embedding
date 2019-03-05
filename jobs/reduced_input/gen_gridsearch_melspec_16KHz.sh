#!/usr/bin/env bash

n_mels=(64 32)
l_hop=(160 320 640) # (10ms 20ms 40ms)

outfile='jobs_l3embedding_train_melspec2_16KHz_reduced_input.sh'

for m in ${n_mels[@]}; do
    for h in ${l_hop[@]}; do
        echo "sbatch l3embedding-train-melspec2-16KHz-reduced-input.sbatch " $m $h >> $outfile
    done
done