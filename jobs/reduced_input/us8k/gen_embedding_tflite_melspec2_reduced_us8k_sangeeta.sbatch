#!/usr/bin/env bash

#SBATCH --job-name=gen-embedding-tflite-melspec2-us8k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="gen-embedding-tflite-melspec2-us8k-%A-%a.out"
#SBATCH --err="gen-embedding-tflite-melspec2-us8k-%A-%a.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
echo 'Using tensorflow 1.14.0 CPU version'
#module load cudnn/10.0v7.4.2.24
    
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

SRCDIR=.
L3_MODEL_PATH=$SCRATCH/quantization/${weight_dir}/${tflite_model}
US8K_PATH=/beegfs/jtc440/UrbanSound8K
METADATA_PATH=$US8K_PATH/metadata/UrbanSound8K.csv
DATA_DIR=$US8K_PATH/audio
OUTPUT_DIR=$SCRATCH/embeddings/features/us8k/quantized_l3/${QUANT_MODE}/${weight_dir}
DATASET='us8k'

echo "ASR=$ASR"
echo "NUM MELS=$NUM_MELS"
echo "HOP LENGTH=$HOP_LEN"
echo "DFT=$NUM_DFT"
echo "WEIGHT FILE=$L3_MODEL_PATH"

if [[ ${#SPLITS[@]} -eq 10 ]]
then
    FMAX=${SPLITS[-1]}
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

