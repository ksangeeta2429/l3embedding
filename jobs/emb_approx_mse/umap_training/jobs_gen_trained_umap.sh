#!/usr/bin/env bash

sbatch gen_trained_umap_emb.sbatch train umap_ndata=1500000_emb=256_nbrs=5_mindist=0.3_mtrc=correlation.sav 102400
sbatch gen_trained_umap_emb.sbatch valid umap_ndata=1500000_emb=256_nbrs=5_mindist=0.3_mtrc=correlation.sav 102400

sbatch gen_trained_umap_emb.sbatch train umap_ndata=1000000_emb=256_nbrs=5_mindist=0.3_mtrc=correlation.sav 102400
sbatch gen_trained_umap_emb.sbatch valid umap_ndata=1000000_emb=256_nbrs=5_mindist=0.3_mtrc=correlation.sav 102400