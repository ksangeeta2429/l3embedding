# Min_dist = 0.3
sbatch generate_save_reduced_emb.sbatch 1000 0.3
sleep 3
sbatch generate_save_reduced_emb.sbatch 10000 0.3
sleep 3
sbatch generate_save_reduced_emb.sbatch 100000 0.3

#Min_dist = 0.5
sbatch generate_save_reduced_emb.sbatch 1000 0.5
sleep 3
sbatch generate_save_reduced_emb.sbatch 10000 0.5
sleep 3
sbatch generate_save_reduced_emb.sbatch 100000 0.5