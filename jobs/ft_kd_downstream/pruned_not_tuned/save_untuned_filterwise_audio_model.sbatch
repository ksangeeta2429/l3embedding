#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=l3-save-filterwise-audio-model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=1-0
#SBATCH --mail-type=FAIL
#SBATCH --output="l3-save-filterwise-audio-model-%j.out"
#SBATCH --err="l3-save-filterwise-audio-model-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`
source activate l3embedding-new-cpu

module purge
if [ $USER == "sk7898" ]; then
  module load cuda/8.0.44
  module load cudnn/8.0v6.0
else
  module load cudnn/9.0v7.3.0.29
fi

SRCDIR=.
WEIGHT_PATH=$SRCDIR/models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5
TRAIN_DATA_DIR=/beegfs/work/AudioSetSamples/music_train
VAL_DATA_DIR=/beegfs/work/AudioSetSamples/music_valid
OUTPUT_DIR=$SRCDIR/pruned_model/
NUM_GPUS=1
NUM_FILTERS=( "$@" )

python $SRCDIR/03_train_pruning.py \
    --model-type cnn_L3_melspec2 \
    --validation-epoch-size 1024 \
    --validation-batch-size 64 \
    --checkpoint-interval 10 \
    --num_filters ${NUM_FILTERS[*]} \
    --gpus $NUM_GPUS \
    --filterwise \
    --save-model \
    --random-state 20180216 \
    --verbose \
    $WEIGHT_PATH \
    $TRAIN_DATA_DIR \
    $VAL_DATA_DIR \
    $OUTPUT_DIR
