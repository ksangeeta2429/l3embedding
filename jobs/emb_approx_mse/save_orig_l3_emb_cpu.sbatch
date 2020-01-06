#!/usr/bin/env bash

#SBATCH --job-name=save-orig-l3-embedding
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=7-0
<<<<<<< HEAD
#SBATCH --mail-user=sk7898@nyu.edu
=======
>>>>>>> defd94fb3e3d7fa5274b072226a2781c88ec6389
#SBATCH --mail-type=ALL
#SBATCH --output="save-orig-l3-embedding-%j.out"
#SBATCH --err="save-orig-l3-embedding-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
if [ $USER == "sk7898" ]; then
    source activate l3embedding-tf-14-cpu
else
<<<<<<< HEAD
    source activate l3embedding-cpu
=======
    source activate l3embedding-new-cpu
>>>>>>> defd94fb3e3d7fa5274b072226a2781c88ec6389
fi

SRCDIR=.
DATASET=$1
OUT_TYPE=$2
<<<<<<< HEAD

if ([ "$DATASET" = "train" ]); then
   DATA_DIR=/beegfs/work/AudioSetSamples/music_train
   OUTPUT_DIR=/scratch/sk7898/orig_l3_embeddings/music_train

elif([ "$DATASET" = "valid" ]); then
   DATA_DIR=/beegfs/work/AudioSetSamples/music_valid
   OUTPUT_DIR=/scratch/sk7898/orig_l3_embeddings/music_valid
=======
PARTITION_NUM=$3

if ([ "$DATASET" = "train" ]); then
   DATA_DIR=/beegfs/work/AudioSetSamples/music_train
   OUTPUT_DIR=$SCRATCH/orig_l3_embeddings/music_train

elif([ "$DATASET" = "valid" ]); then
   DATA_DIR=/beegfs/work/AudioSetSamples/music_valid
   OUTPUT_DIR=$SCRATCH/orig_l3_embeddings/music_valid
>>>>>>> defd94fb3e3d7fa5274b072226a2781c88ec6389
else
   echo "First argument can only be train/valid"
   exit
fi

<<<<<<< HEAD
python $SRCDIR/02_save_orig_l3_emb.py \
      --random-state 20180216 \
      --out-type $OUT_TYPE \
      $DATA_DIR \
      $OUTPUT_DIR \
=======
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
>>>>>>> defd94fb3e3d7fa5274b072226a2781c88ec6389
