#!/usr/bin/env bash

#SBATCH --job-name=gen-reduced-embedding
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="gen-reduced-embedding-neighbor-10-%j.out"
#SBATCH --err="gen-reduced-embedding-neighbor-10-%j.err"

source ~/.bashrc
source activate l3embedding-cpu
cd `git rev-parse --show-toplevel`

module purge

SRCDIR=.
DATASET=$1
BATCH_SIZE=$2
EMB_LEN=$3

APPROX_MODE="umap"
NEIGHBORS_LIST=( 10 )
METRIC_LIST=( 'correlation' ) 
MIN_DIST_LIST=( 0.3 0.5 )
#TSNE_ITER_LIST=( 500 )

if ([ "$DATASET" = "train" ]); then
   DATA_DIR=$SCRATCH/orig_l3_embeddings/music_train
   OUTPUT_DIR=$SCRATCH/reduced_embeddings/music_train_${BATCH_SIZE}_${NEIGHBORS_LIST[0]}
   #For testing
   #DATA_DIR=$SCRATCH/temp_data
   #OUTPUT_DIR=$SCRATCH/reduced_embeddings
   echo $OUTPUT_DIR

elif([ "$DATASET" = "valid" ]); then
   DATA_DIR=$SCRATCH/orig_l3_embeddings/music_valid
   OUTPUT_DIR=$SCRATCH/reduced_embeddings/music_valid_${BATCH_SIZE}_${NEIGHBORS_LIST[0]}

else
   echo "First argument can only be train/valid"
   exit
fi

#alias nogpu='export CUDA_VISIBLE_DEVICES=-1;'

if ([ "$APPROX_MODE" = "umap" ]); then
  echo "UMAP Approximation"
  python $SRCDIR/02_generate_save_reduced_emb.py \
      --batch-size $BATCH_SIZE \
      --approx-mode $APPROX_MODE \
      --neighbors-list ${NEIGHBORS_LIST[*]} \
      --min-dist-list ${MIN_DIST_LIST[*]} \
      --metric-list ${METRIC_LIST[*]} \
      --random-state 20180216 \
      $DATA_DIR \
      $OUTPUT_DIR \
      $EMB_LEN

elif([ "$APPROX_MODE" = "tsne" ]); then
  echo "t-SNE Approximation"
  python $SRCDIR/02_generate_save_reduced_emb.py \
      --batch-size $BATCH_SIZE \
      --approx-mode $APPROX_MODE \
      --neighbors-list ${NEIGHBORS_LIST[*]} \
      --tsne-iter-list ${TSNE_ITER_LIST[*]} \
      --metric-list ${METRIC_LIST[*]} \
      --random-state 20180216 \
      $DATA_DIR \
      $OUTPUT_DIR \
      $EMB_LEN
else
    echo 'Approximation mode can only be UMAP or t-SNE'
    exit
fi
