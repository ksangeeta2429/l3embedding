#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=gen-embedding-pruned-melspec2-us8k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=FAIL
#SBATCH --output="gen-embedding-pruned-melspec2-us8k-%A-%a.out"
#SBATCH --err="gen-embedding-pruned-melspec2-us8k-%A-%a.err"

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
#L3_MODEL_PATH=${filename}
L3_MODEL_PATH=$SCRATCH/l3pruning/embedding/fixed/${filename}
#ORIG_L3_MODEL_PATH=$SRCDIR/models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5
L3_MODEL_TYPE='cnn_L3_melspec2_masked'
L3_POOLING_TYPE='short' #'original'
US8K_PATH=/beegfs/jtc440/UrbanSound8K
METADATA_PATH=$US8K_PATH/metadata/UrbanSound8K.csv
DATA_DIR=$US8K_PATH/audio
OUTPUT_DIR=$SCRATCH/embeddings
DATASET='us8k'
#THRESHOLD=(0.0 0.017678589 0.012369654 0.010109939 0.004976296 0.006389169 0.005634685 0.0067510903) # Command line args list
THRESHOLD=( "$@" )
#GRP='audio_model'

echo Model path: $L3_MODEL_PATH
echo Thresholds: ${THRESHOLD[@]}

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
    --thresholds ${THRESHOLD[*]} \
    --hop-size 0.1 \
    --gpus 0 \
    --us8k-metadata-path $METADATA_PATH \
    $DATASET \
    $DATA_DIR \
    $OUTPUT_DIR
