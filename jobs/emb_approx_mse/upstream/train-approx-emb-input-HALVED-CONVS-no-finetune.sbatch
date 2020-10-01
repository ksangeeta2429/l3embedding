#!/usr/bin/env bash

#SBATCH --gres=gpu:2
#SBATCH --job-name=train-emb-approx-no-ft
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="train-emb-approx-no-ft-%j.out"
#SBATCH --err="train-emb-approx-no-ft-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
module load cudnn/9.0v7.3.0.29
if [ $USER == "sk7898" ]; then
    source activate l3embedding-tf-12
else
    source activate l3embedding-new-cpu
fi

SRCDIR=. 
TRAIN_DATA_DIR=/beegfs/work/AudioSetSamples/music_train
VAL_DATA_DIR=/beegfs/work/AudioSetSamples/music_valid
OUTPUT_DIR=$SCRATCH/l3pruning/embedding_approx_mse
GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo' # REPLACE THIS
NUM_GPUS=2

EMBEDDING_DIR=$1
ASR=$2
NUM_MELS=$3
HOP_LEN=$4
DFT=$5

echo "SAMPLING RATE=$ASR"
echo "NUM MELS=$NUM_MELS"
echo "HOP LENGTH=$HOP_LEN"
echo "DFT=$DFT"

EMBEDDING_BASE=${EMBEDDING_DIR##*/}
IFS='_' 
read -ra SPLITS <<< "$EMBEDDING_BASE" 
APPROX_MODE='umap'
APPROX_TRAIN_SIZE="$(cut -d'=' -f2 <<<"${SPLITS[1]}")" 
NEIGHBORS="$(cut -d'=' -f2 <<<"${SPLITS[3]}")"
MIN_DIST="$(cut -d'=' -f2 <<<"${SPLITS[4]}")"
METRIC="$(cut -d'=' -f2 <<<"${SPLITS[5]}")"
IFS=' '

echo "APPROXIMATION=${APPROX_MODE}"
echo "UMAP_TRAIN_DATA=${APPROX_TRAIN_SIZE}"
echo "NEIGHBORS=${NEIGHBORS}"
echo "MIN_DIST=${MIN_DIST}"
echo "METRIC=${METRIC}"

EMBEDDING_TRAIN_DIR=${EMBEDDING_DIR}/music_train
EMBEDDING_VALID_DIR=${EMBEDDING_DIR}/music_valid

if [ $# -gt 5 ]
then
  FMAX=$6
  echo "FMAX=$FMAX"

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
    --approx-train-size $APPROX_TRAIN_SIZE \
    --neighbors $NEIGHBORS \
    --min-dist $MIN_DIST \
    --metric $METRIC \
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
    --verbose \
    $TRAIN_DATA_DIR \
    $VAL_DATA_DIR \
    $EMBEDDING_TRAIN_DIR \
    $EMBEDDING_VALID_DIR \
    $OUTPUT_DIR
else
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
    --approx-train-size $APPROX_TRAIN_SIZE \
    --neighbors $NEIGHBORS \
    --min-dist $MIN_DIST \
    --metric $METRIC \
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
    $EMBEDDING_TRAIN_DIR \
    $EMBEDDING_VALID_DIR \
    $OUTPUT_DIR
fi
