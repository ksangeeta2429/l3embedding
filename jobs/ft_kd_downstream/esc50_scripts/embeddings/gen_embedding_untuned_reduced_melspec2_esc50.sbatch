#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=gen-embedding-reduced-melspec2-esc50
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2-0
#SBATCH --mail-type=ALL
#SBATCH --output="gen-embedding-reduced-melspec2-esc50-%A-%a.out"
#SBATCH --err="gen-embedding-reduced-melspec2-esc50-%A-%a.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

if [ $USER == "sk7898" ]; then
    source activate l3embedding-new-cpu
else
    source activate l3embedding-new-cpu
fi

module purge
module load cudnn/9.0v7.3.0.29

# Process args
filename=$1
shift

#SRCDIR=$HOME/dev/l3embedding
SRCDIR=$HOME/l3embedding
#L3_MODEL_PATH=${filename}
L3_MODEL_PATH=$SRCDIR/pruned_model/${filename}
#ORIG_L3_MODEL_PATH=$SRCDIR/models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5
L3_MODEL_TYPE='cnn_L3_melspec2_reduced_audioonly'
L3_POOLING_TYPE='short' #'original'
DATA_DIR=$SCRATCH/data/esc50/audio
OUTPUT_DIR=$SCRATCH/embeddings
DATASET='esc50'
NUM_FILTERS=( "$@" ) # Command line args list


#module purge
#module load cuda/8.0.44
#module load cudnn/8.0v6.0
#module load ffmpeg/intel/3.2.2

python $SRCDIR/05_generate_embedding_samples.py \
    --random-state 20180302 \
    --verbose \
    --features 'l3comp' \
    --l3embedding-model-path $L3_MODEL_PATH \
    --l3embedding-model-type $L3_MODEL_TYPE \
    --l3embedding-pooling-type $L3_POOLING_TYPE \
    --num_filters ${NUM_FILTERS[*]} \
    --hop-size 0.1 \
    --gpus 0 \
    $DATASET \
    $DATA_DIR \
    $OUTPUT_DIR
