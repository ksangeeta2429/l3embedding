#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=gen-embedding-reduced-melspec2-sonyc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2-0
#SBATCH --mail-type=ALL
#SBATCH --output="gen-embedding-reduced-melspec2-sonyc-%A-%a.out"
#SBATCH --err="gen-embedding-reduced-melspec2-sonyc-%A-%a.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`
source activate l3embedding-new-cpu

module purge
if [ $USER == "sk7898" ]; then
    module load cudnn/10.0v7.4.2.24
else
    module load cudnn/9.0v7.3.0.29
fi

# Process args
weight_file=$1
weight_dir="$(cut -d'.' -f1 <<<"$weight_file")"

IFS='_'
read -ra SPLITS <<< "$weight_dir"
ASR=${SPLITS[3]}
NUM_MELS=${SPLITS[4]}
HOP_LEN=${SPLITS[5]}
NUM_DFT=${SPLITS[6]}
IFS=' '

#L3_POOLING_TYPE=$(echo $ASR | sed 's/0*$//')'K_'${NUM_MELS}'_'${HOP_LEN}'_'${NUM_DFT}

echo "ASR=$ASR"
#echo "POOL SIZE=$L3_POOLING_TYPE"
echo "NUM MELS=$NUM_MELS"
echo "HOP LENGTH=$HOP_LEN"
echo "DFT=$NUM_DFT"

SRCDIR=.
#L3_MODEL_PATH=${filename}
L3_MODEL_PATH=$SCRATCH/l3pruning/embedding/fixed/reduced_input/${weight_file} #$SCRATCH/l3pruning/embedding/fixed/reduced_input/${weight_file}
#L3_MODEL_TYPE='cnn_L3_melspec2'
#L3_POOLING_TYPE='original'
SONYC_UST_PATH=beegfs/work/sonyc/ust #/beegfs/jtc440/sonyc_ust
DATA_DIR=$SONYC_UST_PATH/audio/48k #$SONYC_UST_PATH/data
METADATA_PATH=$SONYC_UST_PATH/annotations/v0.4/annotations.csv
OUTPUT_DIR=$SCRATCH/embeddings
DATASET='sonyc_ust'

echo "WEIGHT FILE=$L3_MODEL_PATH"

if [[ $weight_file == *"fmax"* ]]
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

