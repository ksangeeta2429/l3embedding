#!/usr/bin/env bash

NEIGHBORS_LIST=( 50 100 200 300 )
MIN_DIST_LIST=( 0.3 0.5 )

for nbr in ${NEIGHBORS_LIST[@]}; do
    for dst in ${MIN_DIST_LIST[@]}; do
        echo sbatch generate_save_reduced_emb.sbatch $nbr $dst >> jobs_generate_save_reduced_emb.sh
    done
done