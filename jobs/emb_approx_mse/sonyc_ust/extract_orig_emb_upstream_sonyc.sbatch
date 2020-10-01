#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=extract-reduced-emb-sonyc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2-0
#SBATCH --mail-type=ALL
#SBATCH --output="extract-reduced-emb-sonyc-%A-%a.out"
#SBATCH --err="extract-reduced-emb-sonyc-%A-%a.err"

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

#L3_POOLING_TYPE=$(echo $ASR | sed 's/0*$//')'K_'${NUM_MELS}'_'${HOP_LEN}'_'${NUM_DFT}

echo "ASR=$ASR"
echo "NUM_MELS=$NUM_MELS"
echo "HOP_LENGTH=$HOP_LEN"
echo "DFT=$NUM_DFT"

SRCDIR=.
L3_MODEL_PATH=$SCRATCH/embedding_approx_mse/models/sonyc/mse_original/${WEIGHT_FILE}/model_best_valid_loss.h5
SONYC_UST_PATH=/beegfs/dr2915/sonyc_ust
DATA_DIR=$SONYC_UST_PATH/data
METADATA_PATH=$DATA_DIR/annotations.csv
OUTPUT_DIR=$SCRATCH/embedding_approx_mse
DATASET='sonyc_ust'

echo "WEIGHT FILE=$L3_MODEL_PATH"

if [[ $WEIGHT_FILE == *"fmax_None"* ]]
then
    echo "No explicit FMAX"
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
        --sonyc-ust-annotation-path $METADATA_PATH \
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
        --save-raw \
        --samp-rate $ASR \
        --num-mels $NUM_MELS \
        --mel-hop-length $HOP_LEN \
        --num-dft $NUM_DFT \
        --freq-max $FMAX \
        --with-melSpec \
        --sonyc-ust-annotation-path $METADATA_PATH \
        $DATASET \
        $DATA_DIR \
        $OUTPUT_DIR
fi

