# Trains -- umap_ndata=1024000_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean
sbatch gen_trained_umap_emb.sbatch train umap_ndata=1024000_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean.sav 102400 0
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=1024000_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean.sav 102400 1
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=1024000_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean.sav 102400 2
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=1024000_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean.sav 102400 3
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=1024000_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean.sav 102400 4
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=1024000_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean.sav 102400 5
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=1024000_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean.sav 102400 6
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=1024000_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean.sav 102400 7
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=1024000_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean.sav 102400 8
sleep 5
sbatch gen_trained_umap_emb.sbatch train umap_ndata=1024000_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean.sav 102400 9

# Valids -- umap_ndata=1024000_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean
sbatch gen_trained_umap_emb.sbatch valid umap_ndata=1024000_emb=256_nbrs=1000_mindist=0.3_mtrc=euclidean.sav 102400
