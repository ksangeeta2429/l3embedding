#!/usr/bin/env bash

#SBATCH --gres=gpu:4
#SBATCH --job-name=l3embedding-train-melspec2-reduced-input-Sangeeta
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="l3embedding-train-melspec2-reduced-input-%j.out"
#SBATCH --err="l3embedding-train-melspec2-reduced-input-%j.err"

source ~/.bashrc
source activate l3embedding-new-cpu
cd `git rev-parse --show-toplevel`

SRCDIR=. #/home/sk7898/l3embedding
WEIGHT_DIR=$SCRATCH/l3pruning/embedding/fixed/reduced_input
TRAIN_DATA_DIR=/beegfs/work/AudioSetSamples/music_train
VAL_DATA_DIR=/beegfs/work/AudioSetSamples/music_valid
OUTPUT_DIR=$SCRATCH/distillation
GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo' # REPLACE THIS
NUM_GPUS=4
TEACHER_WEIGHT_PATH=$WEIGHT_DIR/$1
STUDENT_WEIGHT_PATH=$WEIGHT_DIR/$2
LOSS_TYPE=$3

module purge
if [ $USER == "sk7898" ]; then
  module load cuda/8.0.44
  module load cudnn/8.0v6.0
else
  module load cudnn/9.0v7.3.0.29
fi

python $SRCDIR/03_train_knowledge_distilled_student.py \
    --num-epochs 300 \
    --train-epoch-size 4096 \
    --train-batch-size 64 \
    --model-type cnn_L3_melspec2 \
    --validation-epoch-size 1024 \
    --validation-batch-size 64 \
    --checkpoint-interval 10 \
    --gpus $NUM_GPUS \
    --loss-type $LOSS_TYPE \
    --teacher-weight-path $TEACHER_WEIGHT_PATH \
    --student-weight-path $STUDENT_WEIGHT_PATH \
    --learning-rate 0.00001 \
    --random-state 20180216 \
    --gsheet-id $GSHEET_ID \
    --google-dev-app-name $GOOGLE_DEV_APP_NAME \
    --verbose \
    $TRAIN_DATA_DIR \
    $VAL_DATA_DIR \
    $OUTPUT_DIR
