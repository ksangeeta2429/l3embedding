#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=calibrate-quant-model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --output="calibrate-quant-model-%j.out"
#SBATCH --err="calibrate-quant-model-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

SRCDIR=.

module purge
module load cudnn/10.1v7.6.5.32

echo 'Using tensorflow 2.0 GPU'
source activate l3embedding-tf-2-gpu

python $SRCDIR/calibrate_quant_model.py
