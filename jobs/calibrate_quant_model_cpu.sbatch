#!/usr/bin/env bash

#SBATCH --job-name=calibrate-quant-model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --output="calibrate-quant-model-%j.out"
#SBATCH --err="calibrate-quant-model-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

SRCDIR=.

module purge
echo 'Using tensorflow 2.0 CPU'
source activate l3embedding-tf-2-cpu

python $SRCDIR/calibrate_quant_model.py
