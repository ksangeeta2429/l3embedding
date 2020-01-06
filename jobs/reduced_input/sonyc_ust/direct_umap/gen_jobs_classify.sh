#!/usr/bin/env bash

NEIGHBORS_LIST=( 25 50 75 )
MIN_DIST_LIST=( 0.3 0.5 0.7 )
OUTFILE=jobs_classify_umap_gpu.sh #jobs_direct_umap_classify_l3_ust.sh
for nbr in "${NEIGHBORS_LIST[@]}"; do
    for dst in "${MIN_DIST_LIST[@]}"; do
        echo sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=${nbr}_mindist=${dst}_mtrc=euclidean >> $OUTFILE
        echo sleep 3 >> $OUTFILE
        echo sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=${nbr}_mindist=${dst}_mtrc=euclidean 1 128 >> $OUTFILE
        echo sleep 3 >> $OUTFILE
        echo sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=${nbr}_mindist=${dst}_mtrc=euclidean 1 256 >> $OUTFILE
        echo sleep 3 >> $OUTFILE
        echo sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=${nbr}_mindist=${dst}_mtrc=euclidean 2 128 >> $OUTFILE
        echo sleep 3 >> $OUTFILE
        echo sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=${nbr}_mindist=${dst}_mtrc=euclidean 2 256 >> $OUTFILE
        echo sleep 3 >> $OUTFILE
        #echo "\n" >> $OUTFILE
    done
done
