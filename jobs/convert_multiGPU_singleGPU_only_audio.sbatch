#!/usr/bin/env bash

#SBATCH --gres=gpu:4
#SBATCH --job-name=l3embedding-train-melspec2-reduced-input
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="l3embedding-convert-multiGPU-singleGPU-%j.out"
#SBATCH --err="l3embedding-convert-multiGPU-singleGPU-%j.err"

source ~/.bashrc
source activate l3embedding-new-cpu
cd `git rev-parse --show-toplevel`

SRCDIR=.

module purge
if [ $USER == "sk7898" ]; then
  module load cuda/8.0.44
  module load cudnn/8.0v6.0
else
  module load cudnn/9.0v7.3.0.29
fi

MODEL_PATH=$1
SAMPLING_RATE=$2
NUM_MELS=$3
HOP_LEN=$4
DFT=$5
OUTPUT_DIR=$SCRATCH/l3pruning/embedding/fixed/reduced_input 

python $SRCDIR/04_convert_multiGPU_singleGPU.py \
    --multiGPU-weight-dir $MODEL_PATH \
    --samp-rate $SAMPLING_RATE \
    --num-mels $NUM_MELS \
    --hop-length $HOP_LEN \
    --num-dft $DFT \
    --only-audio \
    $OUTPUT_DIR
