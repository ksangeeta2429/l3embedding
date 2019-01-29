#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=l3-pruning-layers-melspec2-modelonly
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=7-0
#SBATCH --mail-type=FAIL
#SBATCH --output="l3-pruning-layers-melspec2-modelonly-%j.out"
#SBATCH --err="l3-pruning-layers-melspec2-modelonly-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`
source activate l3embedding-new-cpu

module purge
module load cudnn/9.0v7.3.0.29

SRCDIR=.
WEIGHT_PATH=$SRCDIR/models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5
TRAIN_DATA_DIR=/beegfs/work/AudioSetSamples/music_train
VAL_DATA_DIR=/beegfs/work/AudioSetSamples/music_valid
OUTPUT_DIR=$SRCDIR/pruned_model/
NUM_GPUS=1
SPARSITY=( "$@" ) # Command line args list

python $SRCDIR/03_train_pruning.py \
    --model-type cnn_L3_melspec2 \
    --validation-epoch-size 1024 \
    --validation-batch-size 64 \
    --checkpoint-interval 10 \
    --sparsity ${SPARSITY[*]} \
    --gpus $NUM_GPUS \
    --random-state 20180216 \
    --verbose \
    $WEIGHT_PATH \
    $TRAIN_DATA_DIR \
    $VAL_DATA_DIR \
    $OUTPUT_DIR