#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=classifier-train-array-us8k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=1-0
#SBATCH --mail-type=FAIL
#SBATCH --output="classifier-train-array-us8k-%A-%a.out"
#SBATCH --err="classifier-train-array-us8k-%A-%a.err"


source ~/.bashrc
cd `git rev-parse --show-toplevel`

if [ $USER == "sk7898" ]; then
    source activate l3embedding-tf-12
else
    source activate l3embedding-new-cpu
fi

SRCDIR=.
FEATURES_DIR=$1
OUTPUT_DIR=$SCRATCH/embeddings/embedding_approx_mse
MODEL_TYPE='mlp'
FEATURE_MODE='framewise'
GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo'
FOLD_NUM=$2

module purge
module load cudnn/9.0v7.3.0.29

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