#!/usr/bin/env bash

#SBATCH --job-name=gen-trained-embedding
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150GB
#SBATCH --time=3-0
#SBATCH --mail-type=FAIL
#SBATCH --output="gen-trained-embedding-%j.out"
#SBATCH --err="gen-trained-embedding-%j.err"

source ~/.bashrc
source activate l3embedding-cpu
cd `git rev-parse --show-toplevel`

module purge

SRCDIR=.
DATASET=$1
MODEL_FILE=$2
MODEL_PATH=/scratch/dr2915/reduced_embeddings/umap/models/music/$MODEL_FILE
BATCH_SIZE=$3

echo DATASET=$DATASET
echo MODEL_PATH=$MODEL_PATH

OUT_DIR="${MODEL_FILE/.sav/}"

if ([ "$DATASET" = "train" ]); then
   DATA_DIR=/scratch/sk7898/orig_l3_embeddings/music_train
   OUTPUT_DIR=$SCRATCH/reduced_embeddings/umap/$OUT_DIR/music_train
   echo OUTPUT_DIR=$OUTPUT_DIR

elif([ "$DATASET" = "valid" ]); then
   echo "Validation Data!"
   DATA_DIR=/scratch/sk7898/orig_l3_embeddings/music_valid
   OUTPUT_DIR=$SCRATCH/reduced_embeddings/umap/$OUT_DIR/music_valid
   echo OUTPUT_DIR=$OUTPUT_DIR
else
   echo "First argument can only be train/valid"
   exit
fi

#alias nogpu='export CUDA_VISIBLE_DEVICES=-1;'

if [ $# -gt 3 ]; then
    PARTITION_NUM=$4
    echo "PARTITION_NUM=$PARTITION_NUM"
    python -u $SRCDIR/02_extract_trained_umap_emb.py \
      --random-state 20180216 \
      --batch-size $BATCH_SIZE \
      --partition-num $PARTITION_NUM \
      $MODEL_PATH \
      $DATA_DIR \
      $OUTPUT_DIR \
      256
else
    python -u $SRCDIR/02_extract_trained_umap_emb.py \
      --random-state 20180216 \
      --batch-size $BATCH_SIZE \
      $MODEL_PATH \
      $DATA_DIR \
      $OUTPUT_DIR \
      256
fi
