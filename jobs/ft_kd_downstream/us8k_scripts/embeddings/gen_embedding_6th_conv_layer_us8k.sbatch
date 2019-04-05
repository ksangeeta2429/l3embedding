#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=gen-embedding-6th-conv-layer-us8k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=FAIL
#SBATCH --output="gen-embedding-6th-conv-layer-us8k-%A-%a.out"
#SBATCH --err="gen-embedding-6th-conv-layer-us8k-%A-%a.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

if [ $USER == "sk7898" ]; then
    source activate l3embedding-new-cpu
else
    source activate l3embedding-cpu
fi

# Process args
filename=$1
shift

#SRCDIR=$HOME/dev/l3embedding
SRCDIR=.
#L3_MODEL_PATH=${filename}
L3_MODEL_PATH=$SCRATCH/l3pruning/embedding/fixed/${filename}
L3_MODEL_TYPE='cnn_L3_melspec2'
L3_POOLING_TYPE='short' #'original'
FROM_CONV_LAYER=6
US8K_PATH=/beegfs/jtc440/UrbanSound8K
METADATA_PATH=$US8K_PATH/metadata/UrbanSound8K.csv
DATA_DIR=$US8K_PATH/audio
OUTPUT_DIR=$SCRATCH/embeddings
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
