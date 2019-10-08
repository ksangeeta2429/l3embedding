#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=gen-embedding-reduced-melspec2-us8k
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
echo 'Using tensorflow 1.12'
module load cudnn/9.0v7.3.0.29
    
if [ $USER == "sk7898" ]; then
  source activate l3embedding-tf-12
else
  source activate l3embedding-new-cpu
fi

# Process args
WEIGHT_DIR=$1

MODEL_DESC="$(cut -d'/' -f8 <<<"${WEIGHT_DIR}")"

IFS='_'
read -ra SPLITS <<< "$MODEL_DESC"
ASR="${SPLITS[0]}" 
NUM_MELS="${SPLITS[1]}" 
HOP_LEN="${SPLITS[2]}" 
NUM_DFT="${SPLITS[3]}"
IFS=' '

echo "ASR=$ASR"
echo "NUM MELS=$NUM_MELS"
echo "HOP LENGTH=$HOP_LEN"
echo "DFT=$NUM_DFT"

SRCDIR=.
#L3_MODEL_PATH=${filename}
L3_MODEL_PATH=${WEIGHT_DIR}/model_best_valid_loss.h5 
US8K_PATH=/beegfs/jtc440/UrbanSound8K
METADATA_PATH=$US8K_PATH/metadata/UrbanSound8K.csv
DATA_DIR=$US8K_PATH/audio
OUTPUT_DIR=$SCRATCH/embeddings/embedding_approx_mse
DATASET='us8k'

echo "WEIGHT FILE=$L3_MODEL_PATH"

if [ ${#SPLITS[@]} -eq 7 and ${SPLITS[6]} != "None" ]
then
    FMAX="${SPLITS[6]}"
    echo "FMAX=$FMAX"
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
        --us8k-metadata-path $METADATA_PATH \
        $DATASET \
        $DATA_DIR \
        $OUTPUT_DIR
fi

