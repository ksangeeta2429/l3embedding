#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=save_2hr_relevance
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --output="save_2hr_relevance-%j.out"
#SBATCH --err="save_2hr_relevance-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

SRCDIR=.

module purge
module load cudnn/9.0v7.3.0.29

echo 'Using tensorflow 1.12.0 GPU'
source activate l3embedding-tf-12-gpu

python $SRCDIR/save_2hr_relevance.py
