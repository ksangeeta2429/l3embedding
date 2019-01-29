#!/usr/bin/env bash

#SBATCH --job-name=gen-embedding-reduced-melspec2-esc50
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="gen-embedding-reduced-melspec2-esc50-%A-%a.out"
#SBATCH --err="gen-embedding-reduced-melspec2-esc50-%A-%a.err"


source ~/.bashrc
cd `git rev-parse --show-toplevel`

if [ $USER == "sk7898" ]; then
    source activate l3embedding-new-cpu
else
    source activate l3embedding-cpu
fi

# Process args
filename=$1
shift

#SRCDIR=$HOME/dev/l3embedding
SRCDIR=.
L3_MODEL_PATH=$SCRATCH/l3pruning/embedding/fixed/${filename}
L3_POOLING_TYPE='short'
L3_MODEL_TYPE='cnn_L3_melspec2_reduced'
DATA_DIR=$SCRATCH/data/esc50/audio
OUTPUT_DIR=$SCRATCH/embeddings
DATASET='esc50'
#FILTERS=( 64 48 64 64 64 64 128 128 )
FILTERS=( "$@" )

echo Model path: $L3_MODEL_PATH
echo Num Filters: ${FILTERS[@]}

module purge
#module load cuda/8.0.44
#module load cudnn/8.0v6.0
module load ffmpeg/intel/3.2.2

python $SRCDIR/05_generate_embedding_samples.py \
    --random-state 20180302 \
    --verbose \
    --features 'l3comp' \
    --l3embedding-model-path $L3_MODEL_PATH \
    --l3embedding-model-type $L3_MODEL_TYPE \
    --l3embedding-pooling-type $L3_POOLING_TYPE \
    --num_filters ${FILTERS[*]} \
    --hop-size 0.1 \
    --gpus 0 \
    $DATASET \
    $DATA_DIR \
    $OUTPUT_DIR
