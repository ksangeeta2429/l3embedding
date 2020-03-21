#!/usr/bin/env bash

#SBATCH --gres=gpu:4
#SBATCH --job-name=quant-aware-train-same-conv
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --output="quant-aware-same-conv-%j.out"
#SBATCH --err="quant-aware-same-conv-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

SRCDIR=. #/home/sk7898/l3embedding
TRAIN_DATA_DIR=/beegfs/work/AudioSetSamples_environmental/environmental_train #/beegfs/work/AudioSetSamples/music_train
VAL_DATA_DIR=/beegfs/work/AudioSetSamples_environmental/environmental_valid #/beegfs/work/AudioSetSamples/music_valid
OUTPUT_DIR=$SCRATCH/models/reduced_input
GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo' # REPLACE THIS
NUM_GPUS=4
ASR=$1
NUM_MELS=$2
HOP_LEN=$3
DFT=$4
MODEL_DIRECTORY=$5

echo "SAMPLING RATE=$ASR"
echo "NUM MELS=$NUM_MELS"
echo "HOP LENGTH=$HOP_LEN"
echo "DFT=$DFT"
echo "MODEL_DIRECTORY=$MODEL_DIRECTORY"

module purge
module load cudnn/9.0v7.3.0.29
if [ $USER == "sk7898" ]; then
    #module load cudnn/10.0v7.6.2.24
    source activate l3embedding-tf-12-gpu
else
    source activate l3embedding-new-cpu
fi

python $SRCDIR/03_train_quant_embedding.py \
    --num-epochs 5 \
    --train-epoch-size 4096 \
    --train-batch-size 32 \
    --model-type cnn_L3_melspec2 \
    --validation-epoch-size 1024 \
    --validation-batch-size 32 \
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
    --model-dir $MODEL_DIRECTORY \
    --verbose \
    $TRAIN_DATA_DIR \
    $VAL_DATA_DIR \
    $OUTPUT_DIR
