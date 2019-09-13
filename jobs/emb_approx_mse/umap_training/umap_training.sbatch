#!/usr/bin/env bash

#SBATCH --job-name=train-embedding
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=250GB
#SBATCH --time=7-0
#SBATCH --mail-type=FAIL
#SBATCH --output="train-embedding-%j.out"
#SBATCH --err="train-embedding-%j.err"

source ~/.bashrc
source activate l3embedding-cpu
cd `git rev-parse --show-toplevel`

module purge

SRCDIR=.
DATASET=$1
BATCH_SIZE=$2
EMB_LEN=$3
NEIGHBORS=$4
METRIC='correlation'
MIN_DIST=$5

echo DATASET=$DATASET
echo BATCH_SIZE=$BATCH_SIZE
echo EMB_LEN=$EMB_LEN
echo NEIGHBORS=$NEIGHBORS
echo METRIC=$METRIC
echo MIN_DIST=$MIN_DIST

if ([ "$DATASET" = "music" ]); then
   DATA_DIR=/scratch/sk7898/orig_l3_embeddings/music_train
   OUTPUT_DIR=$SCRATCH/reduced_embeddings/umap/models
   #For testing
   #DATA_DIR=/scratch/sk7898/temp_data
   #OUTPUT_DIR=/scratch/sk7898/reduced_embeddings
   echo OUTPUT_DIR=$OUTPUT_DIR

elif([ "$DATASET" = "sonyc" ]); then
   DATA_DIR=/scratch/sk7898/orig_l3_embeddings/sonyc_train
   OUTPUT_DIR=OUTPUT_DIR=$SCRATCH/reduced_embeddings/umap/models

else
   echo "First argument can only be music/sonyc"
   exit
fi

#alias nogpu='export CUDA_VISIBLE_DEVICES=-1;'

python -u $SRCDIR/02_train_umap_emb.py \
  --batch-size $BATCH_SIZE \
  --neighbors $NEIGHBORS \
  --min-dist $MIN_DIST \
  --metric $METRIC \
  --random-state 20180216 \
  $DATA_DIR \
  $OUTPUT_DIR \
  $EMB_LEN
