#!/usr/bin/env bash

#SBATCH --job-name=gen-reduced-embedding
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="gen-reduced-embedding-neighbor-5-%j.out"
#SBATCH --err="gen-reduced-embedding-neighbor-5-%j.err"

source ~/.bashrc
source activate l3embedding-cpu
cd `git rev-parse --show-toplevel`

module purge

SRCDIR=.
BATCH_SIZE=300000
EMB_LEN=256

APPROX_MODE="umap"
NEIGHBORS_LIST=( $1 ) #( 50 100 200 300 )
METRIC_LIST=( 'euclidean' )
MIN_DIST_LIST=( $2 ) #( 0.3 0.5 )

DATA_DIR=/scratch/dr2915/embeddings/features/sonyc_ust/l3/openl3/l3-mel256-music-512
OUTPUT_DIR=$SCRATCH/reduced_embeddings/music_train_${BATCH_SIZE}_${NEIGHBORS_LIST[0]}
OUTPUT_DIR=$SCRATCH/reduced_embeddings/umap/umap_direct_ust/umap_emb=256_nbrs=${NEIGHBORS_LIST[0]}_mindist=${MIN_DIST_LIST[0]}_mtrc=${METRIC_LIST[0]}
#For testing
#DATA_DIR=$SCRATCH/temp_data
#OUTPUT_DIR=$SCRATCH/reduced_embeddings
echo "DATA_DIR: "$DATA_DIR
echo "OUTPUT_DIR: "$OUTPUT_DIR

#alias nogpu='export CUDA_VISIBLE_DEVICES=-1;'

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
