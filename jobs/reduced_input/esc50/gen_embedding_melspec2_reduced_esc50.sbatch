#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=gen-emb-reduced-esc50
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2-0
#SBATCH --mail-type=ALL
#SBATCH --output="esc50-gen-emb-%A-%a.out"
#SBATCH --err="esc50-gen-embgen-emb-%A-%a.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
echo 'Using tensorflow 1.12'
module load cudnn/9.0v7.3.0.29
    
if [ $USER == "sk7898" ]; then
  source activate l3embedding-tf-12
else
  source activate l3embedding-new-cpu
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
DATASET='esc50'
L3_MODEL_PATH=$SCRATCH/l3pruning/embedding/fixed/reduced_input/${weight_file} 
L3_MODEL_TYPE='cnn_L3_melspec2'
DATA_DIR=/beegfs/sk7898/esc50/audio
OUTPUT_DIR=$SCRATCH/embeddings

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


