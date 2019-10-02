#!/usr/bin/env bash

#sbatch gen_trained_umap_emb.sbatch train umap_ndata=1500000_emb=256_nbrs=5_mindist=0.3_mtrc=correlation.sav 102400
#sbatch gen_trained_umap_emb.sbatch valid umap_ndata=1500000_emb=256_nbrs=5_mindist=0.3_mtrc=correlation.sav 102400

#sbatch gen_trained_umap_emb.sbatch train umap_ndata=1000000_emb=256_nbrs=5_mindist=0.3_mtrc=correlation.sav 102400
#sbatch gen_trained_umap_emb.sbatch valid umap_ndata=1000000_emb=256_nbrs=5_mindist=0.3_mtrc=correlation.sav 102400

# ICASSP jobs -- music

sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=10_mindist=0.3_mtrc=euclidean.sav 102400
sbatch gen_trained_umap_emb.sbatch valid umap_ndata=2048000_emb=256_nbrs=10_mindist=0.3_mtrc=euclidean.sav 102400

sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=20_mindist=0.3_mtrc=euclidean.sav 102400
sbatch gen_trained_umap_emb.sbatch valid umap_ndata=2048000_emb=256_nbrs=20_mindist=0.3_mtrc=euclidean.sav 102400

sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=30_mindist=0.3_mtrc=euclidean.sav 102400
sbatch gen_trained_umap_emb.sbatch valid umap_ndata=2048000_emb=256_nbrs=30_mindist=0.3_mtrc=euclidean.sav 102400

# Valids -- umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean
#       and umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean
sbatch gen_trained_umap_emb.sbatch valid umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400
sbatch gen_trained_umap_emb.sbatch valid umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400

# Trains -- umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 0
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 1
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 2
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 3
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 4
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 5
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 6
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 7
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 8
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 9
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 10
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 11
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 12
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 13
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 14
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 15
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 16
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 17
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 18
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=40_mindist=0.3_mtrc=euclidean.sav 102400 19

# Trains -- umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 0
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 1
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 2
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 3
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 4
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 5
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 6
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 7
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 8
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 9
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 10
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 11
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 12
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 13
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 14
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 15
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 16
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 17
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 18
sleep 1
sbatch gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400 19