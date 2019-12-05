#!/usr/bin/env bash

#SBATCH --job-name=train-embedding
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=400GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="train-embedding-%j.out"
#SBATCH --err="train-embedding-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge

if [ $USER == "sk7898" ]; then
  source activate l3embedding-tf-14-cpu
else
  source activate l3embedding-cpu
fi

SRCDIR=.
DATASET=$1
BATCH_SIZE=$2
EMB_LEN=$3
NEIGHBORS=$4
MIN_DIST=$5

echo DATASET=$DATASET
echo BATCH_SIZE=$BATCH_SIZE
echo EMB_LEN=$EMB_LEN
echo NEIGHBORS=$NEIGHBORS
echo METRIC=euclidean
echo MIN_DIST=$MIN_DIST

if ([ "$DATASET" = "music" ]); then
   DATA_DIR=/beegfs/dr2915/umap/training_data/music/umap_training_ndata=$BATCH_SIZE
   OUTPUT_DIR=$SCRATCH/reduced_embeddings/umap/models/music
   #For testing
   #DATA_DIR=/scratch/sk7898/temp_data
   #OUTPUT_DIR=/scratch/sk7898/reduced_embeddings
elif([ "$DATASET" = "sonyc" ]); then
   DATA_DIR=/scratch/dr2915/reduced_embeddings/umap/training_data/sonyc/umap_training_ndata=$BATCH_SIZE
   OUTPUT_DIR=$SCRATCH/reduced_embeddings/umap/models/sonyc
else
   echo "First argument can only be music/sonyc"
   exit
fi
echo OUTPUT_DIR=$OUTPUT_DIR
#alias nogpu='export CUDA_VISIBLE_DEVICES=-1;'

python -u $SRCDIR/02_train_umap_emb.py \
  --batch-size $BATCH_SIZE \
  --neighbors $NEIGHBORS \
  --min-dist $MIN_DIST \
  --random-state 20180216 \
  $DATA_DIR \
  $OUTPUT_DIR \
  $EMB_LEN
