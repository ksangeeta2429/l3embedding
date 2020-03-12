#!/usr/bin/env bash

#SBATCH --gres=gpu:4
#SBATCH --job-name=halved-conv-melspec2-continue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --output="halved-conv-melspec2-continue-%j.out"
#SBATCH --err="halved-conv-melspec2-continue-%j.err"

source ~/.bashrc
source activate l3embedding-new-cpu
cd `git rev-parse --show-toplevel`

SRCDIR=. #/home/sk7898/l3embedding
#WEIGHT_PATH=$SRCDIR/models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5
TRAIN_DATA_DIR=/beegfs/work/AudioSetSamples_environmental/environmental_train #/beegfs/work/AudioSetSamples/music_train
VAL_DATA_DIR=/beegfs/work/AudioSetSamples_environmental/environmental_valid #/beegfs/work/AudioSetSamples/music_valid
OUTPUT_DIR=$SCRATCH/models/reduced_input
GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo' # REPLACE THIS
NUM_GPUS=4
ASR=8000 #$1
NUM_MELS=64 #$2
HOP_LEN=160 #$3
DFT=1024 #$4
CONTINUE_MODEL_DIRECTORY='/scratch/sk7898/models/reduced_input/embedding/environmental/cnn_L3_melspec2/20200222214814' #$5

echo "SAMPLING RATE=$ASR"
echo "NUM MELS=$NUM_MELS"
echo "HOP LENGTH=$HOP_LEN"
echo "DFT=$DFT"
echo "CONTINUE_MODEL_DIRECTORY=$CONTINUE_MODEL_DIRECTORY"

module purge
module load cudnn/9.0v7.3.0.29
source activate l3embedding-tf-12-gpu

if [ $# -gt 5 ]
then
  FMAX=$6
  echo "FMAX=$FMAX"

  python $SRCDIR/03_train_embedding.py \
    --num-epochs 300 \
    --train-epoch-size 4096 \
    --train-batch-size 64 \
    --model-type cnn_L3_melspec2 \
    --validation-epoch-size 1024 \
    --validation-batch-size 64 \
    --checkpoint-interval 10 \
    --gpus $NUM_GPUS \
    --samp-rate $ASR \
    --num-mels $NUM_MELS \
    --hop-length $HOP_LEN \
    --num-dft $DFT \
    --freq-max $FMAX \
    --halved-filters \
    --learning-rate 0.00001 \
    --random-state 20180216 \
    --gsheet-id $GSHEET_ID \
    --google-dev-app-name $GOOGLE_DEV_APP_NAME \
    --continue-model-dir $CONTINUE_MODEL_DIRECTORY \
    --verbose \
    $TRAIN_DATA_DIR \
    $VAL_DATA_DIR \
    $OUTPUT_DIR
else
  python $SRCDIR/03_train_embedding.py \
    --num-epochs 300 \
    --train-epoch-size 4096 \
    --train-batch-size 64 \
    --model-type cnn_L3_melspec2 \
    --validation-epoch-size 1024 \
    --validation-batch-size 64 \
    --checkpoint-interval 10 \
    --gpus $NUM_GPUS \
    --samp-rate $ASR \
    --num-mels $NUM_MELS \
    --hop-length $HOP_LEN \
    --num-dft $DFT \
    --halved-filters \
    --learning-rate 0.00001 \
    --random-state 20180216 \
    --gsheet-id $GSHEET_ID \
    --google-dev-app-name $GOOGLE_DEV_APP_NAME \
    --continue-model-dir $CONTINUE_MODEL_DIRECTORY \
    --verbose \
    $TRAIN_DATA_DIR \
    $VAL_DATA_DIR \
    $OUTPUT_DIR
fi
