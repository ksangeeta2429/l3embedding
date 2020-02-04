#!/usr/bin/env bash

#SBATCH --job-name=dpp-sampling
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=2-0
#SBATCH --mail-type=ALL
#SBATCH --output="dpp-sampling-%j.out"
#SBATCH --err="dpp-sampling-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge

if [ $USER == "sk7898" ]; then
  source activate l3embedding-tf-14-cpu
else
  source activate l3embedding-cpu
fi

python dpp_sampling.py
