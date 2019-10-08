#!/usr/bin/env bash

#SBATCH --job-name=gen-trained-embedding
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150GB
#SBATCH --time=3-0
#SBATCH --mail-type=ALL
#SBATCH --output="gen-trained-embedding-%j.out"
#SBATCH --err="gen-trained-embedding-%j.err"

source ~/.bashrc
source activate l3embedding-cpu
cd `git rev-parse --show-toplevel`

module purge

SRCDIR=.
MODEL_PATH=$1
BATCH_SIZE=$2

echo MODEL_PATH=$MODEL_PATH

MODEL_FILE="${MODEL_PATH##*/}"
OUT_DIR="${MODEL_FILE/.sav/}"

DATA_DIR=/scratch/mc6591/sonyc_ndata=60028928
echo DATA_DIR=$DATA_DIR
OUTPUT_DIR=$SCRATCH/reduced_embeddings/umap/$OUT_DIR/sonyc
echo OUTPUT_DIR=$OUTPUT_DIR

if [ $# -gt 2 ]; then
    PARTITION_NUM=$3
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
