#!/usr/bin/env bash

#SBATCH --gres=gpu:2
#SBATCH --job-name=l3-pruning-blocks-layers-melspec2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=1-0
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=roy.174@osu.edu
#SBATCH --output="l3-pruning-blocks-layers-melspec2-%j.out"
#SBATCH --err="l3-pruning-blocks-layers-melspec2-%j.err"

source ~/.bashrc
cd $SCRATCH/l3embedding/l3embedding
source activate l3embedding-new-cpu

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0

python pruning.py
