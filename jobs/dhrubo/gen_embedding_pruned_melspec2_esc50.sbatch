#!/usr/bin/env bash

#SBATCH --job-name=gen-embedding-pruned-melspec2-esc50
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="gen-embedding-pruned-melspec2-esc50-%A-%a.out"
#SBATCH --err="gen-embedding-pruned-melspec2-esc50-%A-%a.err"

cd `git rev-parse --show-toplevel`

if [ $USER == "sk7898" ]; then
    source activate l3embedding-new-cpu
else
    source activate l3embedding-cpu
fi


SRCDIR=.
L3_MODEL_PATH=${filename}
#L3_MODEL_PATH=''
L3_POOLING_TYPE='short'
L3_MODEL_TYPE='cnn_L3_melspec2_masked'
DATA_DIR=/scratch/hhw230/data/esc50/audio
OUTPUT_DIR=/scratch/sk7898/embeddings
DATASET='esc50'
THRESHOLD=( "$@" ) # Command line args list
#THRESHOLD=(0.0 0.017678589 0.012369654 0.010109939 0.004976296 0.006389169 0.005634685 0.0067510903) 

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
    --fold $SLURM_ARRAY_TASK_ID \
    $DATASET \
    $DATA_DIR \
    $OUTPUT_DIR
