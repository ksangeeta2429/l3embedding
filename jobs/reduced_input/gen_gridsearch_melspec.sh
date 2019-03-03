#!/usr/bin/env bash

n_mels=(256 128 64 32)
l_hop=(242 484 968 1936) # (5ms 10ms 20ms 40ms)

outfile='jobs_l3embedding_train_melspec2_reduced_input.sh'

for m in ${n_mels[@]}; do
    for h in ${l_hop[@]}; do
        echo "sbatch l3embedding-train-melspec2-reduced-input.sbatch " $m $h >> $outfile
    done
done