#!/usr/bin/env bash

NEIGHBORS_LIST=( 50 100 200 300 )
MIN_DIST_LIST=( 0.3 0.5 )

for nbr in "${NEIGHBORS_LIST[@]}"; do
    for dst in "${MIN_DIST_LIST[@]}"; do
        echo sbatch direct_umap_classify_l3_ust.sbatch music umap_emb=256_nbrs=${nbr}_mindist=${dst}_mtrc=euclidean >> jobs_direct_umap_classify_l3_ust.sh
        echo sleep 3 >> jobs_direct_umap_classify_l3_ust.sh
        echo sbatch direct_umap_classify_l3_ust.sbatch music umap_emb=256_nbrs=${nbr}_mindist=${dst}_mtrc=euclidean 1 128 >> jobs_direct_umap_classify_l3_ust.sh
        echo sleep 3 >> jobs_direct_umap_classify_l3_ust.sh
        echo sbatch direct_umap_classify_l3_ust.sbatch music umap_emb=256_nbrs=${nbr}_mindist=${dst}_mtrc=euclidean 1 256 >> jobs_direct_umap_classify_l3_ust.sh
        echo sleep 3 >> jobs_direct_umap_classify_l3_ust.sh
        echo sbatch direct_umap_classify_l3_ust.sbatch music umap_emb=256_nbrs=${nbr}_mindist=${dst}_mtrc=euclidean 2 128 >> jobs_direct_umap_classify_l3_ust.sh
        echo sleep 3 >> jobs_direct_umap_classify_l3_ust.sh
        echo sbatch direct_umap_classify_l3_ust.sbatch music umap_emb=256_nbrs=${nbr}_mindist=${dst}_mtrc=euclidean 2 256 >> jobs_direct_umap_classify_l3_ust.sh
        echo sleep 3 >> jobs_direct_umap_classify_l3_ust.sh
        echo -e "\n"
    done
done