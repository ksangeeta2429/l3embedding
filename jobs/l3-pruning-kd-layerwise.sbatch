#!/usr/bin/env bash

#SBATCH --gres=gpu:4
#SBATCH --job-name=l3-pruning-kd-layerwise
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="l3-pruning-kd-layerwise-%j.out"
#SBATCH --err="l3-pruning-kd-layerwise-%j.err"

source ~/.bashrc
source activate l3embedding-new-cpu
cd `git rev-parse --show-toplevel`

SRCDIR=.
WEIGHT_PATH=L3 Model Weight Path (Can be found in Models directory)
TRAIN_DATA_DIR=Training directory path
VAL_DATA_DIR=Validation directory path
OUTPUT_DIR=$SCRATCH/l3pruning
GOOGLE_DEV_APP_NAME=Google Dev App
GSHEET_ID=Google Sheet ID
NUM_GPUS=4
SPARSITY=( "$@" ) # Command line args list

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0


#Sample Run Command: sbatch l3-pruning-kd-layerwise.sbatch 0 70 70 75 60 80 80 85

python $SRCDIR/03_train_pruning.py \
    $WEIGHT_PATH \
    $TRAIN_DATA_DIR \
    $VAL_DATA_DIR \
    $OUTPUT_DIR \
    --num-epochs 300 \
    --train-epoch-size 4096 \
    --train-batch-size 64 \
    --model-type cnn_L3_melspec2 \
    --validation-epoch-size 1024 \
    --validation-batch-size 64 \
    --checkpoint-interval 10 \
    --retrain \
    --layerwise \
    --sparsity ${SPARSITY[*]} \
    --gpus $NUM_GPUS \
    --learning-rate 0.00001 \
    --random-state 20180216 \
    --gsheet-id $GSHEET_ID \
    --google-dev-app-name $GOOGLE_DEV_APP_NAME \
    --verbose