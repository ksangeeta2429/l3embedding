#!/usr/bin/env bash

source ~/.bashrc
source activate l3embedding-new-cpu
cd `git rev-parse --show-toplevel`

SRCDIR=/home/sk7898/l3embedding
WEIGHT_PATH=$SRCDIR/models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5
TRAIN_DATA_DIR=/beegfs/work/AudioSetSamples/music_train
VAL_DATA_DIR=/beegfs/work/AudioSetSamples/music_valid
OUTPUT_DIR=/scratch/sk7898/pruned_model
GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo' # REPLACE THIS
NUM_GPUS=1
SPARSITY=(0 0.3 0.4 0.5 0.3 0.5 0.5 0.6) # Command line args list
FINETUNE=True

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0

python $SRCDIR/03_train_pruning.py \
    --num-epochs 300 \
    --train-epoch-size 4096 \
    --train-batch-size 64 \
    --model-type cnn_L3_melspec2 \
    --validation-epoch-size 1024 \
    --validation-batch-size 64 \
    --checkpoint-interval 10 \
    --retrain \
    --sparsity ${SPARSITY[*]} \
    --gpus $NUM_GPUS \
    --learning-rate 0.00001 \
    --random-state 20180216 \
    --gsheet-id $GSHEET_ID \
    --google-dev-app-name $GOOGLE_DEV_APP_NAME \
    --verbose \
    $WEIGHT_PATH \
    $TRAIN_DATA_DIR \
    $VAL_DATA_DIR \
    $OUTPUT_DIR
