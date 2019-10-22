#!/usr/bin/env bash

# UMAP trained on SONYC
sbatch sonyc_ust_gen_trained_umap_emb.sbatch /scratch/sk7898/reduced_embeddings/umap/models/sonyc/umap_ndata=2048000_emb=256_nbrs=20_mindist=0.3_mtrc=euclidean.sav 102400
sleep 5
sbatch sonyc_ust_gen_trained_umap_emb.sbatch /scratch/sk7898/reduced_embeddings/umap/models/sonyc/umap_ndata=2048000_emb=256_nbrs=10_mindist=0.3_mtrc=euclidean.sav 102400
sleep 5
sbatch sonyc_ust_gen_trained_umap_emb.sbatch /scratch/sk7898/reduced_embeddings/umap/models/sonyc/umap_ndata=2048000_emb=256_nbrs=30_mindist=0.3_mtrc=euclidean.sav 102400
sleep 5
sbatch sonyc_ust_gen_trained_umap_emb.sbatch /scratch/sk7898/reduced_embeddings/umap/models/sonyc/umap_ndata=2048000_emb=256_nbrs=50_mindist=0.3_mtrc=euclidean.sav 102400

# UMAP trained on music
sbatch sonyc_ust_gen_trained_umap_emb.sbatch /scratch/dr2915/reduced_embeddings/umap/models/music/umap_ndata=2048000_emb=256_nbrs=10_mindist=0.3_mtrc=euclidean.sav 102400
sleep 5
sbatch sonyc_ust_gen_trained_umap_emb.sbatch /scratch/dr2915/reduced_embeddings/umap/models/music/umap_ndata=2048000_emb=256_nbrs=30_mindist=0.3_mtrc=euclidean.sav 102400