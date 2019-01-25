#!/usr/bin/env bash

#SBATCH --job-name=gen-embedding-pruned-melspec2-us8k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=FAIL
#SBATCH --output="gen-embedding-reduced-melspec2-us8k-%A-%a.out"
#SBATCH --err="gen-embedding-reduced-melspec2-us8k-%A-%a.err"


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
L3_MODEL_PATH=${filename}
#L3_MODEL_PATH=/scratch/dr2915/l3pruning/embedding/fixed/model_best_valid_accuracy_20190118115452.h5 #$SRCDIR/model_best_valid_accuracy_20190118115452.h5
#ORIG_L3_MODEL_PATH=$SRCDIR/models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5
L3_MODEL_TYPE='cnn_L3_melspec2_reduced'
L3_POOLING_TYPE='short' #'original'
US8K_PATH=/beegfs/jtc440/UrbanSound8K
METADATA_PATH=$US8K_PATH/metadata/UrbanSound8K.csv
DATA_DIR=$US8K_PATH/audio
OUTPUT_DIR=/scratch/sk7898/embeddings
DATASET='us8k'
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
    --us8k-metadata-path $METADATA_PATH \
    $DATASET \
    $DATA_DIR \
    $OUTPUT_DIR
