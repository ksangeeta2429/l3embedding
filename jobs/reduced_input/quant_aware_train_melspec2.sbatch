#!/usr/bin/env bash

#SBATCH --gres=gpu:2
#SBATCH --job-name=quant-aware-train-melspec2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="quant-aware-train-melspec2-%j.out"
#SBATCH --err="quant-aware-train-melspec2-%j.err"

source ~/.bashrc
source activate l3embedding-new-cpu
cd `git rev-parse --show-toplevel`

SRCDIR=.
TRAIN_DATA_DIR=/beegfs/work/AudioSetSamples/music_train
VAL_DATA_DIR=/beegfs/work/AudioSetSamples/music_valid
OUTPUT_DIR=$SCRATCH/models/reduced_input
GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo' # REPLACE THIS
NUM_GPUS=2
ASR=8000
NUM_MELS=64
HOP_LEN=160
DFT=1024

echo "SAMPLING RATE=$ASR"
echo "NUM MELS=$NUM_MELS"
echo "HOP LENGTH=$HOP_LEN"
echo "DFT=$DFT"

module purge
module load cudnn/10.0v7.4.2.24
source activate l3embedding-tf-14-gpu

python $SRCDIR/03_train_quant_embedding.py \
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
    --verbose \
    $TRAIN_DATA_DIR \
    $VAL_DATA_DIR \
    $OUTPUT_DIR
