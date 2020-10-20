#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=gen-emb-reduced-esc50
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2-0
#SBATCH --mail-type=ALL
#SBATCH --output="esc50-gen-emb-%A-%a.out"
#SBATCH --err="esc50-gen-emb-%A-%a.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
echo 'Using tensorflow 1.12'
module load cudnn/9.0v7.3.0.29
    
if [ $USER == "sk7898" ]; then
  source activate l3embedding-tf-12-gpu
else
  source activate l3embedding-new-cpu
fi

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
echo "HOP_LEN=$HOP_LEN"

SRCDIR=.
DATASET='esc50'
L3_MODEL_PATH=$SCRATCH/embedding_approx_mse/models/sonyc/pca/dpp/day/500000/pca_batch_500000_len_128_kernel_linear/${WEIGHT_FILE}/model_best_valid_loss.h5
L3_MODEL_TYPE='cnn_L3_melspec2'
DATA_DIR=/beegfs/sk7898/esc50/audio
OUTPUT_DIR=$SCRATCH/embedding_approx_mse

echo "WEIGHT FILE=$L3_MODEL_PATH"

python -u $SRCDIR/05_generate_embedding_samples.py \
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
    $DATASET \
    $DATA_DIR \
    $OUTPUT_DIR


