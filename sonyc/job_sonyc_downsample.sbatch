#!/bin/bash

#SBATCH --job-name=sonyc-map
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-0
#SBATCH --mem=50GB
#SBATCH --mail-type=ALL
#SBATCH --output="sonyc-map-%A-%a.out"
#SBATCH --err="sonyc-map-%A-%a.err"

module purge
source ~/.bashrc
source activate sonyc-research-data

python utils.py downsample_sonyc_points
