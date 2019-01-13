#!/usr/bin/env bash

#SBATCH --gres=gpu:4
#SBATCH --job-name=l3-pruning-layers-melspec2-finetune
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=FAIL
#SBATCH --output="l3-pruning-layers-melspec2-finetune-%j.out"
#SBATCH --err="l3-pruning-layers-melspec2-finetune-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`
source activate l3embedding-new-cpu

module purge
module load cudnn/9.0v7.3.0.29

SRCDIR=.
WEIGHT_PATH=$SRCDIR/models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5
TRAIN_DATA_DIR=/beegfs/work/AudioSetSamples/music_train
VAL_DATA_DIR=/beegfs/work/AudioSetSamples/music_valid
OUTPUT_DIR=/scratch/dr2915/l3pruning
GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo' # REPLACE THIS
NUM_GPUS=4
SPARSITY=( "$@" ) # Command line args list
FINETUNE=True

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
