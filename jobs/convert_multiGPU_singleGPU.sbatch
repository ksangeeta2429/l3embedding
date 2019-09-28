#!/usr/bin/env bash

#SBATCH --gres=gpu:4
#SBATCH --job-name=l3embedding-convert-multiGPU-singleGPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL
#SBATCH --output="l3embedding-convert-multiGPU-singleGPU-%j.out"
#SBATCH --err="l3embedding-convert-multiGPU-singleGPU-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

SRCDIR=.

module purge
echo 'Using tensorflow 1.12'
module load cudnn/9.0v7.3.0.29
    
if [ $USER == "sk7898" ]; then
  source activate l3embedding-tf-12
else
  source activate l3embedding-new-cpu
fi

#MODEL_DIR=$SCRATCH/l3pruning/reduced_input/embedding/music/cnn_L3_melspec2
MODEL_PATH=$1
SAMPLING_RATE=$2
NUM_MELS=$3
HOP_LEN=$4
DFT=$5
NUM_GPUS=4
OUTPUT_DIR=$SCRATCH/l3pruning/embedding/fixed/reduced_input 

#Pass 6th parameter as 'half'
if ([ $# -gt 5 ] && [ "$6" = "half" ]); then
    python $SRCDIR/04_convert_multiGPU_singleGPU.py \
        --multiGPU-weight-dir $MODEL_PATH \
        --gpus $NUM_GPUS \
        --samp-rate $SAMPLING_RATE \
        --num-mels $NUM_MELS \
        --hop-length $HOP_LEN \
        --num-dft $DFT \
        --halved-filters \
        $OUTPUT_DIR
else
    python $SRCDIR/04_convert_multiGPU_singleGPU.py \
        --multiGPU-weight-dir $MODEL_PATH \
        --gpus $NUM_GPUS \
        --samp-rate $SAMPLING_RATE \
        --num-mels $NUM_MELS \
        --hop-length $HOP_LEN \
        --num-dft $DFT \
        $OUTPUT_DIR
fi
