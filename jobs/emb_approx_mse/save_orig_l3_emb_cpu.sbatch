#!/usr/bin/env bash

#SBATCH --job-name=save-orig-l3-embedding
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-user=sk7898@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output="save-orig-l3-embedding-%j.out"
#SBATCH --err="save-orig-l3-embedding-%j.err"

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
OUT_TYPE=$2

if ([ "$DATASET" = "train" ]); then
   DATA_DIR=/beegfs/work/AudioSetSamples/music_train
   OUTPUT_DIR=/scratch/sk7898/orig_l3_embeddings/music_train

elif([ "$DATASET" = "valid" ]); then
   DATA_DIR=/beegfs/work/AudioSetSamples/music_valid
   OUTPUT_DIR=/scratch/sk7898/orig_l3_embeddings/music_valid
else
   echo "First argument can only be train/valid"
   exit
fi

python $SRCDIR/02_save_orig_l3_emb.py \
      --random-state 20180216 \
      --out-type $OUT_TYPE \
      $DATA_DIR \
      $OUTPUT_DIR \
