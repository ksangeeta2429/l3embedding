#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=ft_quant_aware_cmsis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="ft-quant-aware-cmsis-%j.out"
#SBATCH --err="ft-quant-aware-cmsis-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

SRCDIR=.
module purge
module load cudnn/10.0v7.4.2.24
source activate l3embedding-tf-14-gpu

python $SRCDIR/quantization_aware_ft_tflite.py
