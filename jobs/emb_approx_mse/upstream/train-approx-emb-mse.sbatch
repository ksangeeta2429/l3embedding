#!/usr/bin/env bash

#SBATCH --gres=gpu:2
#SBATCH --job-name=mse-sonyc-approx
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="mse-sonyc-approx-%j.out"
#SBATCH --err="mse-sonyc-approx-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
module load cudnn/9.0v7.3.0.29
source activate l3embedding-tf-12-gpu

SRCDIR=. 
APPROX_MODE='mse'
ASR=16000
NUM_MELS=64
HOP_LEN=160
AUDIO_DIR=$SCRATCH/sonyc_30mil
TRAIN_DATA_DIR=${AUDIO_DIR}/train
VAL_DATA_DIR=${AUDIO_DIR}/validate
EMBEDDING_TRAIN_DIR=$TRAIN_DATA_DIR
EMBEDDING_VALID_DIR=$VAL_DATA_DIR
OUTPUT_DIR=$SCRATCH/embedding_approx_mse/models
GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo'
NUM_GPUS=2

echo "APPROXIMATION=${APPROX_MODE}"
echo "TRAIN_DATA_DIR=${TRAIN_DATA_DIR}"
echo "EMBEDDING_TRAIN_DIR=${EMBEDDING_TRAIN_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

python $SRCDIR/03_train_approx_embedding_mse.py \
	--num-epochs 300 \
	--train-epoch-size 4096 \
	--train-batch-size 64 \
	--model-type cnn_L3_melspec2 \
	--validation-epoch-size 1024 \
	--validation-batch-size 64 \
	--checkpoint-interval 10 \
	--gpus $NUM_GPUS \
	--approx-mode $APPROX_MODE \
	--samp-rate $ASR \
	--num-mels $NUM_MELS \
	--hop-length $HOP_LEN \
	--halved-filters \
	--learning-rate 0.00001 \
	--random-state 20180216 \
	--gsheet-id $GSHEET_ID \
	--google-dev-app-name $GOOGLE_DEV_APP_NAME \
	--verbose \
	$TRAIN_DATA_DIR \
	$VAL_DATA_DIR \
	$EMBEDDING_TRAIN_DIR \
	$EMBEDDING_VALID_DIR \
	$OUTPUT_DIR