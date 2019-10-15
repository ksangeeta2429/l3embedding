#!/usr/bin/env bash

sbatch direct_umap_classify_l3_ust.sbatch umap_ndata=2048000_emb=256_nbrs=10_mindist=0.3_mtrc=euclidean sonyc
sbatch direct_umap_classify_l3_ust.sbatch umap_ndata=2048000_emb=256_nbrs=30_mindist=0.3_mtrc=euclidean sonyc

sbatch direct_umap_classify_l3_ust.sbatch umap_ndata=2048000_emb=256_nbrs=10_mindist=0.3_mtrc=euclidean music
sbatch direct_umap_classify_l3_ust.sbatch umap_ndata=2048000_emb=256_nbrs=30_mindist=0.3_mtrc=euclidean music