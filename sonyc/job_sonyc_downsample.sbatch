#!/bin/bash

#SBATCH --job-name=sonyc-downsample
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-0
#SBATCH --mem=70GB
#SBATCH --mail-type=ALL
#SBATCH --output="sonyc-downsample-%A-%a.out"
#SBATCH --err="sonyc-downsample-%A-%a.err"

module purge
source ~/.bashrc
source activate sonyc-research-data

python utils.py downsample_sonyc_points
