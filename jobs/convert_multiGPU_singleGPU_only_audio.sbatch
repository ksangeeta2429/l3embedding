#!/usr/bin/env bash

#SBATCH --gres=gpu:4
#SBATCH --job-name=l3embedding-convert-multiGPU-singleGPU-audio
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=7:00:00
#SBATCH --mail-type=ALL
#SBATCH --output="l3embedding-convert-multiGPU-singleGPU-audio-%j.out"
#SBATCH --err="l3embedding-convert-multiGPU-singleGPU-audio-%j.err"

source ~/.bashrc
source activate l3embedding-new-cpu
cd `git rev-parse --show-toplevel`

SRCDIR=.

module purge
if [ $USER == "sk7898" ]; then
  module load cuda/8.0.44
  module load cudnn/8.0v6.0
else
  module load cudnn/9.0v7.3.0.29
fi

MODEL_PATH=$1
SAMPLING_RATE=$2
NUM_MELS=$3
HOP_LEN=$4
DFT=$5
NUM_GPUS=4
OUTPUT_DIR=$SCRATCH/l3pruning/embedding/fixed/reduced_input 

#Pass 6th parameter as 'half'
if ([ $# -gt 5 ] && [ "$5" = "half" ]); then
    python $SRCDIR/04_convert_multiGPU_singleGPU.py \
	--multiGPU-weight-dir $MODEL_PATH \
	--gpus $NUM_GPUS \
	--samp-rate $SAMPLING_RATE \
	--num-mels $NUM_MELS \
	--hop-length $HOP_LEN \
	--num-dft $DFT \
	--only-audio \
	--halved-filters \
	$OUTPUT_DIR
else
    python $SRCDIR/04_convert_multiGPU_singleGPU.py \
	--multiGPU-weight-dir $MODEL_PATH \
	--gpus $NUM_GPUS \
	--samp-rate $SAMPLING_RATE \
	--num-mels $NUM_MELS \
	--hop-length $HOP_LEN \
	--num-dft $DFT \
	--only-audio \
	$OUTPUT_DIR
fi 
