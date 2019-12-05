#!/usr/bin/env bash

sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=500_mindist=0.3_mtrc=euclidean
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=500_mindist=0.3_mtrc=euclidean 1 128
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=500_mindist=0.3_mtrc=euclidean 1 256
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=500_mindist=0.3_mtrc=euclidean 2 128
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=500_mindist=0.3_mtrc=euclidean 2 256
sleep 3

sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=700_mindist=0.3_mtrc=euclidean
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=700_mindist=0.3_mtrc=euclidean 1 128
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=700_mindist=0.3_mtrc=euclidean 1 256
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=700_mindist=0.3_mtrc=euclidean 2 128
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=700_mindist=0.3_mtrc=euclidean 2 256
sleep 3

sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean 1 128
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean 1 256
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean 2 128
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean 2 256
sleep 3

sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=300_mindist=0.05_mtrc=euclidean
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=300_mindist=0.05_mtrc=euclidean 1 128
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=300_mindist=0.05_mtrc=euclidean 1 256
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=300_mindist=0.05_mtrc=euclidean 2 128
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=300_mindist=0.05_mtrc=euclidean 2 256
sleep 3

sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=300_mindist=0.1_mtrc=euclidean
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=300_mindist=0.1_mtrc=euclidean 1 128
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=300_mindist=0.1_mtrc=euclidean 1 256
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=300_mindist=0.1_mtrc=euclidean 2 128
sleep 3
sbatch direct_umap_classify_l3_ust.sbatch umap_emb=256_nbrs=300_mindist=0.1_mtrc=euclidean 2 256
sleep 3

