#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=classifier-train-array-us8k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=7-0
#SBATCH --mail-type=FAIL
#SBATCH --output="classifier-train-array-us8k-%A-%a.out"
#SBATCH --err="classifier-train-array-us8k-%A-%a.err"


source ~/.bashrc
cd `git rev-parse --show-toplevel`
source activate l3embedding-new-cpu

#SRCDIR=$HOME/dev/l3embedding
SRCDIR=.
#FEATURES_DIR=/scratch/dr2915/embeddings/features/us8k/l3comp/short/models/cnn_l3_melspec2_recent/model_best_valid_accuracy/
FEATURES_DIR=$1
OUTPUT_DIR=/scratch/sk7898/embeddings
MODEL_TYPE='mlp'
FEATURE_MODE='framewise'
GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo'
#FOLD_NUM=$SLURM_ARRAY_TASK_ID
#FOLD_NUM=1
FOLD_NUM=$2

module purge
module load cudnn/9.0v7.3.0.29
#module load cuda/8.0.44
#module load cudnn/8.0v6.0

python $SRCDIR/06_train_classifier.py \
    --random-state 20171021 \
    --model-type $MODEL_TYPE \
    --feature-mode $FEATURE_MODE \
    --num-epochs 150 \
    --train-batch-size 32 \
    --gsheet-id $GSHEET_ID \
    --google-dev-app-name $GOOGLE_DEV_APP_NAME \
    --verbose \
    $FEATURES_DIR \
    $OUTPUT_DIR \
    $FOLD_NUM
