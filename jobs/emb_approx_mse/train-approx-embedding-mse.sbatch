#!/usr/bin/env bash

#SBATCH --gres=gpu:4
#SBATCH --job-name=l3embedding-train-melspec2-reduced-input-Sangeeta
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="l3embedding-train-melspec2-reduced-input-%j.out"
#SBATCH --err="l3embedding-train-melspec2-reduced-input-%j.err"

source ~/.bashrc
source activate l3embedding-new-cpu
cd `git rev-parse --show-toplevel`

SRCDIR=. #/home/sk7898/l3embedding
WEIGHT_DIR=$SCRATCH/l3pruning/embedding/fixed/reduced_input
#TRAIN_DATA_DIR=/beegfs/work/AudioSetSamples/music_train
TRAIN_DATA_DIR=$SCRATCH/temp_data
#VAL_DATA_DIR=/beegfs/work/AudioSetSamples/music_valid
VAL_DATA_DIR=$SCRATCH/temp_data

EMBEDDING_DIR=$SCRATCH/temp_emb
OUTPUT_DIR=$SCRATCH/l3pruning/embedding_approx_mse
GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo' # REPLACE THIS
NUM_GPUS=4

STUDENT_WEIGHT_PATH=$WEIGHT_DIR/$1 #l3_audio_20190723135620_8000_128_64_256_half_fmax_3500.h5
APPROX_MODE=$2
NEIGHBORS=$3
METRIC=$4

#Sample command:
#sbatch emb_approx_mse.sbatch l3_audio_20190723135620_8000_128_64_256_half_fmax_3500.h5 umap 20 correlation 0.3
module purge
if [ $USER == "sk7898" ]; then
  module load cuda/8.0.44
  module load cudnn/8.0v6.0
else
  module load cudnn/9.0v7.3.0.29
fi


if ([ "$2" = "umap" ]); then
  echo "UMAP Approximation"
  MIN_DIST=$5
  python $SRCDIR/03_train_embedding_approx_mse.py \
	  --num-epochs 1 \
	  --train-epoch-size 2048 \
	  --train-batch-size 1024 \
	  --model-type cnn_L3_melspec2 \
	  --validation-epoch-size 1024 \
	  --validation-batch-size 512 \
	  --checkpoint-interval 10 \
	  --gpus $NUM_GPUS \
	  --approx-mode $APPROX_MODE \
	  --neighbors $NEIGHBORS \
	  --min-dist $MIN_DIST \
	  --metric $METRIC\
	  --student-weight-path $STUDENT_WEIGHT_PATH \
	  --learning-rate 0.00001 \
	  --random-state 20180216 \
	  --gsheet-id $GSHEET_ID \
	  --google-dev-app-name $GOOGLE_DEV_APP_NAME \
	  --verbose \
	  $TRAIN_DATA_DIR \
	  $VAL_DATA_DIR \
	  $EMBEDDING_DIR \
	  $OUTPUT_DIR
  
elif ([ "$2" = "tsne" ]); then
    python $SRCDIR/03_train_embedding_approx_mse.py \
	--num-epochs 1 \
	--train-epoch-size 8192 \
	--train-batch-size 1024 \
	--model-type cnn_L3_melspec2 \
	--validation-epoch-size 1024 \
	--validation-batch-size 1024 \
	--checkpoint-interval 10 \
	--gpus $NUM_GPUS \
	--approx-mode $APPROX_MODE \
	--neighbors $NEIGHBORS \
	--metric $METRIC \
	--student-weight-path $STUDENT_WEIGHT_PATH \
	--learning-rate 0.00001 \
	--random-state 20180216 \
	--gsheet-id $GSHEET_ID \
	--google-dev-app-name $GOOGLE_DEV_APP_NAME \
	--verbose \
	$TRAIN_DATA_DIR \
	$VAL_DATA_DIR \
	$EMBEDDING_DIR \
	$OUTPUT_DIR
fi
