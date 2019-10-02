#!/usr/bin/env bash

sbatch rerun_gen_trained_umap_emb.sbatch train umap_ndata=2048000_emb=256_nbrs=20_mindist=0.3_mtrc=euclidean.sav 102400