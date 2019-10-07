#!/usr/bin/env bash

#SBATCH --gres=gpu:2
#SBATCH --job-name=continue-train-embedding-approx-mse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="continue-train-embedding-approx-mse-%j.out"
#SBATCH --err="continue-train-embedding-approx-mse-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
module load cudnn/9.0v7.3.0.29
if [ $USER == "sk7898" ]; then
    source activate l3embedding-tf-12
else
    source activate l3embedding-new-cpu
fi

SRCDIR=. #/home/sk7898/l3embedding
WEIGHT_DIR=$SCRATCH/l3pruning/embedding/fixed/reduced_input
#TRAIN_DATA_DIR=$SCRATCH/temp_data
#VAL_DATA_DIR=$SCRATCH/temp_data
TRAIN_DATA_DIR=/beegfs/work/AudioSetSamples/music_train
VAL_DATA_DIR=/beegfs/work/AudioSetSamples/music_valid
OUTPUT_DIR=$SCRATCH/l3pruning/embedding_approx_mse
GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo' # REPLACE THIS
NUM_GPUS=2

CONTINUE_MODEL_DIRECTORY=$1
#STUDENT_WEIGHT_PATH=$WEIGHT_DIR/$1 #l3_audio_20190723135620_8000_128_64_256_half_fmax_3500.h5
EMBEDDING_DIR=$2

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
echo "CONTINUE_MODEL_DIRECTORY=$CONTINUE_MODEL_DIRECTORY"

EMBEDDING_TRAIN_DIR=${EMBEDDING_DIR}/music_train
EMBEDDING_VALID_DIR=${EMBEDDING_DIR}/music_valid

# CHOICE of learning-rate, train-epoch-size/steps-per-epoch and batch size
# Smaller learning rates will require more training epochs. Conversely, larger learning rates will require fewer training epochs.
# Smaller batch sizes are better suited to smaller learning rates given the noisy estimate of the error gradient.

# Typically, a grid search involves picking values approximately on a logarithmic scale, 
# e.g., a learning rate taken within the set {.1, .01, 10−3, 10−4 , 10−5}

if ([ "$APPROX_MODE" = "umap" ]); then
  if [ $# -eq 1 ]; then
      APPROX_MODE='mse'
      NEIGHBORS=0
      METRIC='mse'
      MIN_DIST=0.0
  fi
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
	  --learning-rate 0.00001 \
	  --random-state 20180216 \
	  --gsheet-id $GSHEET_ID \
	  --google-dev-app-name $GOOGLE_DEV_APP_NAME \
          --continue-model-dir $CONTINUE_MODEL_DIRECTORY \
	  --verbose \
	  $TRAIN_DATA_DIR \
	  $VAL_DATA_DIR \
	  $EMBEDDING_TRAIN_DIR \
          $EMBEDDING_VALID_DIR \
	  $OUTPUT_DIR
  
elif ([ "$APPROX_MODE" = "tsne" ]); then
    python $SRCDIR/03_train_embedding_approx_mse.py \
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
	--metric $METRIC \
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
