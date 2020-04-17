#!/usr/bin/env bash

#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=gen-samp-umap-ust
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-0
#SBATCH --mail-type=FAIL
#SBATCH --output="gen-samp-umap-ust-%j.out"
#SBATCH --err="gen-samp-umap-ust-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
if [ $USER == "sk7898" ]; then
    module load cudnn/9.0v7.3.0.29
    source activate l3embedding-tf-12
else
    # Needs an NVIDIA GPU > 6.0 compute capability, so no K80s
    source activate rapids  # Create env from envs_yml/rapids.yml
fi

SRCDIR=.
BATCH_SIZE=200000
EMB_LEN=256

APPROX_MODE="umap"
NEIGHBORS_LIST=( $1 ) #( 500 1000 5000 )
METRIC_LIST=( 'euclidean' )
MIN_DIST_LIST=( $2 ) #( 0.3 0.5 0.7 )

SAMPLING=spl
TIMESCALE=day
NDATA=200000

DATA_DIR=/scratch/dr2915/sonyc_samples/$SAMPLING/ndata=$NDATA/timescale=$TIMESCALE
TRANSFORM_DATA_DIR=/scratch/dr2915/embeddings/features/sonyc_ust/l3/melSpec_20200304183233_48000_256_242_2048
#OUTPUT_DIR=$SCRATCH/reduced_embeddings/music_train_${BATCH_SIZE}_${NEIGHBORS_LIST[0]}
OUTPUT_DIR=$SCRATCH/reduced_embeddings/umap/samp_umap_ust/$SAMPLING/ndata=$NDATA/timescale=$TIMESCALE/umap_emb=256_nbrs=${NEIGHBORS_LIST[0]}_mindist=${MIN_DIST_LIST[0]}_mtrc=${METRIC_LIST[0]}
#For testing
#DATA_DIR=$SCRATCH/temp_data
#OUTPUT_DIR=$SCRATCH/reduced_embeddings
echo "DATA_DIR: "$DATA_DIR
echo "OUTPUT_DIR: "$OUTPUT_DIR
echo "TRANSFORM_DATA_DIR:" $TRANSFORM_DATA_DIR

#alias nogpu='export CUDA_VISIBLE_DEVICES=-1;'

python -u $SRCDIR/02_generate_save_reduced_emb.py \
  --batch-size $BATCH_SIZE \
  --approx-mode $APPROX_MODE \
  --transform-data-dir $TRANSFORM_DATA_DIR \
  --neighbors-list ${NEIGHBORS_LIST[*]} \
  --min-dist-list ${MIN_DIST_LIST[*]} \
  --metric-list ${METRIC_LIST[*]} \
  --random-state 20180216 \
  $DATA_DIR \
  $OUTPUT_DIR \
  $EMB_LEN
