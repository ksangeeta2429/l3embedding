#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=h7-test-ust-tflite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --output="h7-test-ust-tflite-%j.out"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

SRCDIR=.

module purge
module load cudnn/10.0v7.4.2.24

echo 'Using tensorflow 2.0 GPU'
source activate l3embedding-tf-2-gpu

python $SRCDIR/test_demo_model_h7_ust.py
