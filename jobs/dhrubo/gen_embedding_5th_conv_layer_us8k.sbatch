#!/usr/bin/env bash

#SBATCH --job-name=gen-embedding-7th-conv-layer-us8k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=FAIL
#SBATCH --output="gen-embedding-7th-conv-layer-us8k-%A-%a.out"
#SBATCH --err="gen-embedding-7th-conv-layer-us8k-%A-%a.err"


source ~/.bashrc
cd `git rev-parse --show-toplevel`
source activate l3embedding-cpu

#SRCDIR=$HOME/dev/l3embedding
SRCDIR=.
L3_MODEL_PATH='models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5'
L3_MODEL_TYPE='cnn_L3_melspec2'
L3_POOLING_TYPE='short' #'original'
FROM_CONV_LAYER=5
US8K_PATH=/beegfs/jtc440/UrbanSound8K
METADATA_PATH=$US8K_PATH/metadata/UrbanSound8K.csv
DATA_DIR=$US8K_PATH/audio
OUTPUT_DIR=/scratch/dr2915/embeddings
DATASET='us8k'

module purge
#module load cuda/8.0.44
#module load cudnn/8.0v6.0
module load ffmpeg/intel/3.2.2

python $SRCDIR/05_generate_embedding_samples.py \
    --random-state 20180302 \
    --verbose \
    --features 'l3comp' \
    --l3embedding-model-path $L3_MODEL_PATH \
    --l3embedding-model-type $L3_MODEL_TYPE \
    --from-conv-layer $FROM_CONV_LAYER \
    --l3embedding-pooling-type $L3_POOLING_TYPE \
    --hop-size 0.1 \
    --gpus 0 \
    --us8k-metadata-path $METADATA_PATH \
    $DATASET \
    $DATA_DIR \
    $OUTPUT_DIR
