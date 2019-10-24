#!/usr/bin/env bash

#SBATCH --job-name=create-umap-dataset
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --time=2-0
#SBATCH --mail-type=FAIL
#SBATCH --output="create-umap-dataset-%j.out"
#SBATCH --err="create-umap-dataset-%j.err"

source ~/.bashrc
source activate l3embedding-cpu
cd `git rev-parse --show-toplevel`

module purge

SRCDIR=.
DATASET=$1
TRAINING_SIZE=$2

echo DATASET=$DATASET
echo TRAINING_SIZE=$TRAINING_SIZE

if ([ "$DATASET" = "music" ]); then
   DATA_DIR=/scratch/sk7898/orig_l3_embeddings/music_train
   OUTPUT_DIR=$BEEGFS/umap/training_data/music/umap_training_ndata=$TRAINING_SIZE
   #For testing
   #DATA_DIR=/scratch/sk7898/temp_data
   #OUTPUT_DIR=/scratch/sk7898/reduced_embeddings

elif([ "$DATASET" = "sonyc" ]); then
   DATA_DIR=/scratch/dr2915/reduced_embeddings/sonyc_files_list.csv
   OUTPUT_DIR=$BEEGFS/umap/training_data/sonyc/umap_training_ndata=$TRAINING_SIZE

else
   echo "First argument can only be music/sonyc"
   exit
fi

echo OUTPUT_DIR=$OUTPUT_DIR
#alias nogpu='export CUDA_VISIBLE_DEVICES=-1;'

python -u $SRCDIR/01_create_umap_training_datasets.py \
  --random-state 20180216 \
  $DATA_DIR \
  $OUTPUT_DIR \
  $TRAINING_SIZE
