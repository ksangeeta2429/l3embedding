#!/usr/bin/env bash

#SBATCH --gres=gpu:4
#SBATCH --job-name=l3-knowledge-distillation-melspec2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=62GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sangeeta.osu@gmail.com
#SBATCH --output="l3-knowledge-distillation-melspec2-%j.out"
#SBATCH --err="l3-knowledge-distillation-melspec2-%j.err"

source ~/.bashrc
cd /home/$USER/l3embedding/l3embedding
source activate l3embedding-new-cpu

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0

python train_kd.py
