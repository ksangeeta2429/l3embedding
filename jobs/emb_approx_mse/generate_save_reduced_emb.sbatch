#!/usr/bin/env bash

#SBATCH --job-name=gen-reduced-embedding
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="gen-reduced-embedding-%j.out"
#SBATCH --err="gen-reduced-embedding-%j.err"

source ~/.bashrc
source activate l3embedding-cpu
cd `git rev-parse --show-toplevel`

module purge

SRCDIR=.
DATASET=$1
APPROX_MODE=$2
BATCH_SIZE=$3
EMB_LEN=$4
NEIGHBORS_LIST=( 10 20 30 )
METRIC_LIST=( 'correlation' 'euclidean' )
MIN_DIST_LIST=( 0.3 0.5 )
TSNE_ITER_LIST=( 500 )

if ([ "$DATASET" = "train" ]); then
   DATA_DIR=/beegfs/work/AudioSetSamples/music_train
   #DATA_DIR=$SCRATCH/temp_data
   OUTPUT_DIR=$SCRATCH/reduced_embeddings/music_train
   #OUTPUT_DIR=$SCRATCH/reduced_embeddings

elif([ "$DATASET" = "valid" ]); then
   DATA_DIR=/beegfs/work/AudioSetSamples/music_valid
   OUTPUT_DIR=$SCRATCH/reduced_embeddings/music_valid
else
   echo "First argument can only be train/valid"
   exit
fi

#alias nogpu='export CUDA_VISIBLE_DEVICES=-1;'

if ([ "$2" = "umap" ]); then
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

elif([ "$2" = "tsne" ]); then
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
