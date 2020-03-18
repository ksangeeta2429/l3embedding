#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=gen-emb-tflite-ust
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="gen-emb-tflite-ust-%A-%a.out"
#SBATCH --err="gen-emb-tflite-ust-%A-%a.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
echo 'Using tensorflow 2.0 GPU version'
    
if [ $USER == "sk7898" ]; then
  module load cudnn/10.1v7.6.5.32
  source activate l3embedding-tf-2-gpu
else
  echo 'If you have tensorflow 2.0 (required) in your environment, please make the changes to the script to activate that'
  exit 0
fi

# Process args
weight_dir=$1 #l3_audio_20200304152812_8000_64_160_1024_half
QUANT_MODE=$2 #default_int8_uint8
#weight_dir="$(cut -d'.' -f1 <<<"$weight_file")"

IFS='_' 
read -ra SPLITS <<< "$weight_dir" 
ASR=${SPLITS[3]}
NUM_MELS=${SPLITS[4]}
HOP_LEN=${SPLITS[5]}
NUM_DFT=${SPLITS[6]}
IFS=' '

tflite_model="quantized_${QUANT_MODE}.tflite"

SRCDIR=.
L3_MODEL_PATH=$SCRATCH/quantization/${weight_dir}/${tflite_model}
SONYC_UST_PATH=/beegfs/dr2915/sonyc_ust
DATA_DIR=$SONYC_UST_PATH
METADATA_PATH=$DATA_DIR/annotations.csv
DATASET='sonyc_ust'
OUTPUT_DIR=$SCRATCH/embeddings/features/sonyc_ust/quantized_l3/${QUANT_MODE}/${weight_dir}

echo "ASR=$ASR"
echo "NUM MELS=$NUM_MELS"
echo "HOP LENGTH=$HOP_LEN"
echo "DFT=$NUM_DFT"
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
