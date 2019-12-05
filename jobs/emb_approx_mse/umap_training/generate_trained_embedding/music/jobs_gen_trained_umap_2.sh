#!/usr/bin/env bash

#sbatch gen_trained_umap_emb.sbatch train umap_ndata=1500000_emb=256_nbrs=5_mindist=0.3_mtrc=correlation.sav 102400
#sbatch gen_trained_umap_emb.sbatch valid umap_ndata=1500000_emb=256_nbrs=5_mindist=0.3_mtrc=correlation.sav 102400

#sbatch gen_trained_umap_emb.sbatch train umap_ndata=1000000_emb=256_nbrs=5_mindist=0.3_mtrc=correlation.sav 102400
#sbatch gen_trained_umap_emb.sbatch valid umap_ndata=1000000_emb=256_nbrs=5_mindist=0.3_mtrc=correlation.sav 102400

# ICASSP jobs -- music

# Trains -- /scratch/dr2915/reduced_embeddings/umap/models/music/umap_ndata=2048000_emb=256_nbrs=300_mindist=0.3_mtrc=euclidean
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=300_mindist=0.3_mtrc=euclidean.sav 102400 0
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=300_mindist=0.3_mtrc=euclidean.sav 102400 1
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=300_mindist=0.3_mtrc=euclidean.sav 102400 2
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=300_mindist=0.3_mtrc=euclidean.sav 102400 3
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=300_mindist=0.3_mtrc=euclidean.sav 102400 4
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=300_mindist=0.3_mtrc=euclidean.sav 102400 5
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=300_mindist=0.3_mtrc=euclidean.sav 102400 6
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=300_mindist=0.3_mtrc=euclidean.sav 102400 7
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=300_mindist=0.3_mtrc=euclidean.sav 102400 8
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=300_mindist=0.3_mtrc=euclidean.sav 102400 9
# Valids -- umap_ndata=2048000_emb=256_nbrs=300_mindist=0.3_mtrc=euclidean
sbatch gen_trained_umap_emb.sbatch valid umap_ndata=2048000_emb=256_nbrs=300_mindist=0.3_mtrc=euclidean.sav 102400
