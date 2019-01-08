#!/usr/bin/env bash

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
source activate l3embedding-cpu

#SRCDIR=$HOME/dev/l3embedding
SRCDIR=.
L3_MODEL_PATH=pruned_model/${filename}
L3_MODEL_TYPE='cnn_L3_melspec2_audioonly'
L3_POOLING_TYPE='short' #'original'
US8K_PATH=/beegfs/jtc440/UrbanSound8K
METADATA_PATH=$US8K_PATH/metadata/UrbanSound8K.csv
DATA_DIR=$US8K_PATH/audio
OUTPUT_DIR=/scratch/dr2915/embeddings
DATASET='us8k'

module purge
#module load cuda/8.0.44
#module load cudnn/8.0v6.0
module load ffmpeg/intel/3.2.2

python $SRCDIR/05_generate_embedding_samples.py \
    --random-state 20180302 \
    --verbose \
    --features 'l3comp' \
    --l3embedding-model-path $L3_MODEL_PATH \
    --l3embedding_model_type $L3_MODEL_TYPE \
    --l3embedding-pooling-type $L3_POOLING_TYPE \
    --hop-size 0.1 \
    --gpus 0 \
    --fold $SLURM_ARRAY_TASK_ID \
    --us8k-metadata-path $METADATA_PATH \
    --dataset_name $DATASET \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR
