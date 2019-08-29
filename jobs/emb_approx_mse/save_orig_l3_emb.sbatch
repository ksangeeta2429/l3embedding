#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=save-orig-l3-embedding
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="save-orig-l3-embedding-%j.out"
#SBATCH --err="save-orig-l3-embedding-%j.err"

source ~/.bashrc
source activate l3embedding-new-cpu
cd `git rev-parse --show-toplevel`

module purge

SRCDIR=.
DATASET=$1

if ([ "$DATASET" = "train" ]); then
   DATA_DIR=/beegfs/work/AudioSetSamples/music_train
   OUTPUT_DIR=$SCRATCH/orig_l3_embeddings/music_train

elif([ "$DATASET" = "valid" ]); then
   DATA_DIR=/beegfs/work/AudioSetSamples/music_valid
   OUTPUT_DIR=$SCRATCH/orig_l3_embeddings/music_valid
else
   echo "First argument can only be train/valid"
   exit
fi

python $SRCDIR/02_save_orig_l3_emb.py \
      --random-state 20180216 \
      $DATA_DIR \
      $OUTPUT_DIR \
