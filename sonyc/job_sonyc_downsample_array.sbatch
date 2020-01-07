#!/bin/bash

#SBATCH --job-name=sonyc-downsample
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-0
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --output="sonyc-downsample-%A-%a.out"
#SBATCH --err="sonyc-downsample-%A-%a.err"

module purge
source ~/.bashrc
source activate sonyc-research-data

python -u utils.py downsample_sonyc_points $SLURM_ARRAY_TASK_ID
