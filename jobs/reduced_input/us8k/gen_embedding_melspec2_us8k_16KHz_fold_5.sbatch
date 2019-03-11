#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=gen-embedding-pruned-melspec2-us8k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2-0
#SBATCH --mail-type=FAIL
#SBATCH --output="gen-embedding-pruned-melspec2-us8k-%A-%a.out"
#SBATCH --err="gen-embedding-pruned-melspec2-us8k-%A-%a.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`
source activate l3embedding-new-cpu

module purge
if [ $USER == "sk7898" ]; then
    module load cuda/8.0.44
    module load cudnn/8.0v6.0
else
    module load cudnn/9.0v7.3.0.29
fi

# Process args
NUM_MELS=$1
HOP_LEN=$2

echo "NUM MELS=$NUM_MELS"
echo "HOP LENGTH=$HOP_LEN"

SRCDIR=.
#L3_MODEL_PATH=${filename}
L3_MODEL_PATH='/scratch/dr2915/l3pruning/reduced_input/embedding/music/cnn_L3_melspec2/20190306112819'
L3_MODEL_TYPE='cnn_L3_melspec2_audioonly'
L3_POOLING_TYPE='16k_64_50' #'original'
US8K_PATH=/beegfs/jtc440/UrbanSound8K
METADATA_PATH=$US8K_PATH/metadata/UrbanSound8K.csv
DATA_DIR=$US8K_PATH/audio
OUTPUT_DIR=$SCRATCH/embeddings
DATASET='us8k'

python $SRCDIR/05_generate_embedding_samples.py \
    --random-state 20180302 \
    --verbose \
    --features 'l3' \
    --l3embedding-model-path $L3_MODEL_PATH \
    --l3embedding-pooling-type $L3_POOLING_TYPE \
    --hop-size 0.1 \
    --gpus 0 \
    --samp-rate 16000 \
    --num-mels $NUM_MELS \
    --hop-length $HOP_LEN \
    --num-dft 1024 \
    --fold 5 \
    --us8k-metadata-path $METADATA_PATH \
    $DATASET \
    $DATA_DIR \
    $OUTPUT_DIR


