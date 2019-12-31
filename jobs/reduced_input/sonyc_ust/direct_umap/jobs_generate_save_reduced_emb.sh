sbatch generate_save_reduced_emb.sbatch 500 0.3
sleep 3
sbatch generate_save_reduced_emb.sbatch 700 0.3
sleep 3
sbatch generate_save_reduced_emb.sbatch 1000 0.3
sleep 3
sbatch generate_save_reduced_emb.sbatch 300 0.05
sleep 3
sbatch generate_save_reduced_emb.sbatch 300 0.1
sleep 3

# Large batch sizes
# Min_dist = 0.3
sbatch --mem=100GB generate_save_reduced_emb.sbatch 1000 0.3
sleep 3
sbatch --mem=100GB generate_save_reduced_emb.sbatch 5000 0.3
sleep 3
sbatch --mem=100GB generate_save_reduced_emb.sbatch 10000 0.3
sleep 3

#Min_dist = 0.5
sbatch --mem=100GB generate_save_reduced_emb.sbatch 1000 0.5
sleep 3
sbatch --mem=100GB generate_save_reduced_emb.sbatch 5000 0.5
sleep 3
sbatch --mem=100GB generate_save_reduced_emb.sbatch 10000 0.5
