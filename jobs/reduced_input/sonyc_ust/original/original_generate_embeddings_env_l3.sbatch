#!/usr/bin/env bash

#SBATCH --gres=gpu:4
#SBATCH --job-name=convert-multiGPU-singleGPU-audio
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=5:00:00
#SBATCH --mail-type=ALL
#SBATCH --output="convert-multiGPU-singleGPU-audio-%j.out"
#SBATCH --err="convert-multiGPU-singleGPU-audio-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

SRCDIR=.

module purge
echo 'Using tensorflow 1.12'
module load cudnn/9.0v7.3.0.29

if [ $USER == "sk7898" ]; then
  source activate l3embedding-tf-12-gpu
else
  source activate l3embedding-new-cpu
fi

MODEL_PATH=/scratch/dr2915/models/reduced_input/embedding/environmental/cnn_L3_melspec2/20200304183233
SAMPLING_RATE=48000
NUM_MELS=256
HOP_LEN=242
DFT=2048
NUM_GPUS=4
OUTPUT_DIR=$SCRATCH/models/openl3/environmental/audio_models

python $SRCDIR/04_convert_multiGPU_singleGPU.py \
	--multiGPU-weight-dir $MODEL_PATH \
	--gpus $NUM_GPUS \
	--samp-rate $SAMPLING_RATE \
	--num-mels $NUM_MELS \
	--hop-length $HOP_LEN \
	--num-dft $DFT \
	--only-audio \
	--melSpec \
	$OUTPUT_DIR

L3_MODEL_PATH=$OUTPUT_DIR/l3_audio_melSpec_20200304183233_48000_256_242_2048.h5
SONYC_UST_PATH=/beegfs/dr2915/sonyc_ust
DATA_DIR=$SONYC_UST_PATH #$SONYC_UST_PATH/data
METADATA_PATH=$SONYC_UST_PATH/annotations.csv
OUTPUT_DIR=$SCRATCH/embeddings
DATASET='sonyc_ust'

python $SRCDIR/05_generate_embedding_samples.py \
        --random-state 20180302 \
        --verbose \
        --features 'l3' \
        --l3embedding-model-path $L3_MODEL_PATH \
        --hop-size 0.1 \
        --gpus 0 \
        --save-raw \
        --samp-rate $SAMPLING_RATE \
        --num-mels $NUM_MELS \
        --mel-hop-length $HOP_LEN \
        --num-dft $DFT \
        --sonyc-ust-annotation-path $METADATA_PATH \
        --with-melSpec \
        $DATASET \
        $DATA_DIR \
        $OUTPUT_DIR
