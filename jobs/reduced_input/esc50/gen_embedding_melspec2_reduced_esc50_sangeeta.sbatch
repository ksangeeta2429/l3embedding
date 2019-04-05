#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=gen-embedding-reduced-melspec2-esc50
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2-0
#SBATCH --mail-type=ALL
#SBATCH --output="gen-embedding-reduced-melspec2-esc50-%A-%a.out"
#SBATCH --err="gen-embedding-reduced-melspec2-esc50-%A-%a.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`
source activate l3embedding-new-cpu

module purge
if [ $USER == "sk7898" ]; then
    module load cuda/8.0.44
    module load cudnn/8.0v6.0
else
    module load cudnn/9.0v7.3.0.29
fi

# Process args
weight_file=$1
ASR=$2
NUM_MELS=$3
HOP_LEN=$4
NUM_DFT=$5

L3_POOLING_TYPE=$(echo $ASR | sed 's/0*$//')'K_'${NUM_MELS}'_'${HOP_LEN}'_'${NUM_DFT}

echo "ASR=$ASR"
echo "POOL SIZE=$L3_POOLING_TYPE"

SRCDIR=.
L3_MODEL_PATH=$SCRATCH/l3pruning/embedding/fixed/reduced_input/${weight_file} 
L3_MODEL_TYPE='cnn_L3_melspec2'
DATA_DIR=$SCRATCH/data/esc50/audio
OUTPUT_DIR=$SCRATCH/embeddings
DATASET='esc50'

echo "WEIGHT FILE=$L3_MODEL_PATH"

python $SRCDIR/05_generate_embedding_samples.py \
    --random-state 20180302 \
    --verbose \
    --features 'l3' \
    --l3embedding-model-type $L3_MODEL_TYPE \
    --l3embedding-model-path $L3_MODEL_PATH \
    --l3embedding-pooling-type $L3_POOLING_TYPE \
    --hop-size 0.1 \
    --gpus 0 \
    --samp-rate $ASR \
    --num-mels $NUM_MELS \
    --hop-length $HOP_LEN \
    --num-dft $NUM_DFT \
    $DATASET \
    $DATA_DIR \
    $OUTPUT_DIR


