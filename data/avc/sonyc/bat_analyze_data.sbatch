#!/usr/bin/env bash

#SBATCH --job-name=sonyc-analyze
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10GB
#SBATCH --time=1-0
#SBATCH --mail-type=FAIL
#SBATCH --output="sonyc-analyze-%j.out"
#SBATCH --err="sonyc-analyze-%j.err"

python -u analyze_data.py

