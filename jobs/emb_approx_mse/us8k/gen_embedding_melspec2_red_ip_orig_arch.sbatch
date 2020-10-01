#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=extract-us8k-reduced-melspec
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2-0
#SBATCH --mail-type=ALL
#SBATCH --output="gen-embedding-reduced-melspec2-us8k-%A-%a.out"
#SBATCH --err="gen-embedding-reduced-melspec2-us8k-%A-%a.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
module load cudnn/9.0v7.3.0.29
source activate l3embedding-tf-12-gpu

# Process args
WEIGHT_FILE=$1
WEIGHT_DIR="$(cut -d'.' -f1 <<<"$WEIGHT_FILE")"

IFS='_'
read -ra SPLITS <<< "$WEIGHT_DIR"
ASR=${SPLITS[0]}
NUM_MELS=${SPLITS[1]}
HOP_LEN=${SPLITS[2]}
NUM_DFT=${SPLITS[3]}
IFS=' '

echo "ASR=$ASR"
echo "NUM_MELS=$NUM_MELS"
echo "HOP_LENGTH=$HOP_LEN"
echo "DFT=$NUM_DFT"

SRCDIR=.
L3_MODEL_PATH=$SCRATCH/embedding_approx_mse/models/sonyc/mse_original/${WEIGHT_FILE}/model_best_valid_loss.h5
US8K_PATH=/beegfs/sk7898/UrbanSound8K
METADATA_PATH=$US8K_PATH/metadata/UrbanSound8K.csv
DATA_DIR=$US8K_PATH/audio
OUTPUT_DIR=$SCRATCH/embeddings/embedding_approx_mse
DATASET='us8k'

echo "WEIGHT FILE=$L3_MODEL_PATH"

if [[ $WEIGHT_FILE == *"fmax_None"* ]]
then
    echo "No explicit FMAX"
    python $SRCDIR/05_generate_embedding_samples.py \
        --random-state 20180302 \
        --verbose \
        --features 'l3' \
        --l3embedding-model-path $L3_MODEL_PATH \
        --hop-size 0.1 \
        --gpus 0 \
        --samp-rate $ASR \
        --num-mels $NUM_MELS \
        --mel-hop-length $HOP_LEN \
        --num-dft $NUM_DFT \
        --with-melSpec \
        --us8k-metadata-path $METADATA_PATH \
        $DATASET \
        $DATA_DIR \
        $OUTPUT_DIR
else
    python $SRCDIR/05_generate_embedding_samples.py \
        --random-state 20180302 \
        --verbose \
        --features 'l3' \
        --l3embedding-model-path $L3_MODEL_PATH \
        --hop-size 0.1 \
        --gpus 0 \
        --samp-rate $ASR \
        --num-mels $NUM_MELS \
        --mel-hop-length $HOP_LEN \
        --num-dft $NUM_DFT \
        --freq-max $FMAX \
        --with-melSpec \
        --us8k-metadata-path $METADATA_PATH \
        $DATASET \
        $DATA_DIR \
        $OUTPUT_DIR
fi

