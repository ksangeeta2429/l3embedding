#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=generate-embs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --output="generate-embs-%j.out"
#SBATCH --err="generate-embs-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

SRCDIR=.

module purge
echo 'Using tensorflow 2'
module load cuda/10.2.89 #module load cudnn/10.1v7.6.5.32
source activate l3embedding-tf-2-gpu

VERSION='v2.2'
SAMPLING_RATE=48000
NUM_MELS=256
HOP_LEN=242
DFT=2048
EMBS_DIR=$SCRATCH/sonyc_output/models/openl3/environmental/audio_models
L3_MODEL_PATH=$EMBS_DIR/l3_audio_melSpec_20200304183233_48000_256_242_2048.h5
SONYC_UST_PATH=/scratch/work/sonyc/sonyc/ust
DATA_DIR=$SONYC_UST_PATH/audio/48k
METADATA_PATH=$SONYC_UST_PATH/annotations/$VERSION/annotations.csv
OUTPUT_DIR=$SCRATCH/sonyc_output/embeddings/$VERSION
DATASET='sonyc_ust'

python $SRCDIR/05_generate_embedding_samples.py \
        --random-state 20180302 \
        --verbose \
        --features 'l3' \
        --l3embedding-model-path $L3_MODEL_PATH \
        --hop-size 0.1 \
        --gpus 0 \
        --samp-rate $SAMPLING_RATE \
        --num-mels $NUM_MELS \
        --mel-hop-length $HOP_LEN \
        --num-dft $DFT \
        --sonyc-ust-annotation-path $METADATA_PATH \
        --with-melSpec \
        $DATASET \
        $DATA_DIR \
        $OUTPUT_DIR
