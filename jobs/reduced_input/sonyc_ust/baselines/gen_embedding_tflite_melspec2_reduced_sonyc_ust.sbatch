#!/usr/bin/env bash

#SBATCH --job-name=gen-embedding-tflite-melspec2-sonyc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="gen-embedding-tflite-melspec2-sonyc-%A-%a.out"
#SBATCH --err="gen-embedding-tflite-melspec2-sonyc-%A-%a.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
echo 'Using tensorflow 1.14.0 CPU version'
    
if [ $USER == "sk7898" ]; then
  source activate l3embedding-tf-14-cpu
else
  echo 'If you have tensorflow 1.14.0 (required) in your environment, please make the changes to the script to activate that'
  exit 0
fi

# Process args
weight_file=$1
QUANT_MODE=$2
weight_dir="$(cut -d'.' -f1 <<<"$weight_file")"

IFS='_' 
read -ra SPLITS <<< "$weight_dir" 
ASR=${SPLITS[3]}
NUM_MELS=${SPLITS[4]}
HOP_LEN=${SPLITS[5]}
NUM_DFT=${SPLITS[6]}
IFS=' '

tflite_model="quantized_model_${QUANT_MODE}.tflite"

echo "ASR=$ASR"
echo "NUM MELS=$NUM_MELS"
echo "HOP LENGTH=$HOP_LEN"
echo "DFT=$NUM_DFT"

SRCDIR=.
L3_MODEL_PATH=$SCRATCH/quantization/${weight_dir}/${tflite_model}
SONYC_UST_PATH=/beegfs/jtc440/sonyc_ust
DATA_DIR=$SONYC_UST_PATH/data
METADATA_PATH=$DATA_DIR/annotations.csv
DATASET='sonyc_ust'
OUTPUT_DIR=$SCRATCH/embeddings/features/sonyc_ust/quantized_l3/${QUANT_MODE}/${weight_dir}

echo "WEIGHT FILE=$L3_MODEL_PATH"

if [[ ${#SPLITS[@]} -eq 10 ]]
then
    FMAX=${SPLITS[-1]}
    echo "FMAX=$FMAX"

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
        --freq-max $FMAX \
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
        --samp-rate $ASR \
        --num-mels $NUM_MELS \
        --mel-hop-length $HOP_LEN \
        --num-dft $NUM_DFT \
        --sonyc-ust-annotation-path $METADATA_PATH \
        $DATASET \
        $DATA_DIR \
        $OUTPUT_DIR
fi
