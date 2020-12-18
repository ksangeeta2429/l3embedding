#!/usr/bin/env bash

#SBATCH --job-name=test-val-sens
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --output="test-val-sens-%j.out"
#SBATCH --err="test-val-sens-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
source activate sonyc-research-data

python val_robustness.py
