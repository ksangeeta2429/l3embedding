#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=save-orig-l3-embedding
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="save-orig-l3-embedding-%j.out"
#SBATCH --err="save-orig-l3-embedding-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
module load cudnn/9.0v7.3.0.29
if [ $USER == "sk7898" ]; then
    source activate l3embedding-tf-12
else
    source activate l3embedding-new-cpu
fi

SRCDIR=.
DATASET=$1
OUT_TYPE=$2
PARTITION_NUM=$3

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

if [ $# -gt 3 ]; then
    START_IDX=$4
    echo "START_IDX=$START_IDX"
    python $SRCDIR/02_save_orig_l3_emb.py \
	--random-state 20180216 \
	--start-idx $START_IDX \
	--partition-num $PARTITION_NUM \
	--out-type $OUT_TYPE \
	$DATA_DIR \
	$OUTPUT_DIR
else
    python $SRCDIR/02_save_orig_l3_emb.py \
	--random-state 20180216 \
	--partition-num $PARTITION_NUM \
	--out-type $OUT_TYPE \
	$DATA_DIR \
	$OUTPUT_DIR
fi
