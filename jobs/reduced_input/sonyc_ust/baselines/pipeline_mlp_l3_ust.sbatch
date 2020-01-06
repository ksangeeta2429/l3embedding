#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=pipeline-ust-mlp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=1-0
#SBATCH --mail-type=FAIL
#SBATCH --output="pipeline-ust-mlp-%j.out"

source ~/.bashrc
cd `git rev-parse --show-toplevel`
source activate l3embedding-new-cpu

module purge
module load cudnn/9.0v7.3.0.29

SRCDIR=.
ANNPATH=/beegfs/dr2915/sonyc_ust/annotations.csv
TAXPATH=/beegfs/dr2915/sonyc_ust/dcase-ust-taxonomy.yaml
EMBDIR=$SCRATCH/embeddings/features/sonyc_ust/l3/reduced_input/$1
OUTDIR=$SCRATCH/embeddings/classifier/sonyc_ust/pipeline_ds/mlp

#module purge
#module load sox/intel/14.4.2
#module load ffmpeg/intel/3.2.2
#module load rubberband/intel/1.8.1

if [ $# -gt 1 ]; then
    NUM_HIDDEN=$2
    HIDDEN_SIZE=$3
else
    NUM_HIDDEN=0
    HIDDEN_SIZE=0
fi

EXPDIR=$OUTDIR/$1/${NUM_HIDDEN}_${HIDDEN_SIZE}
OUTPUTDIR=$EXPDIR/results

echo "NUM_HIDDEN=$NUM_HIDDEN"
echo "HIDDEN_SIZE=$HIDDEN_SIZE"

GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo'

python classifier/sonyc_ust/classify.py $ANNPATH $TAXPATH $EMBDIR $EXPDIR results \
    --hidden_layer_size $HIDDEN_SIZE \
    --num_hidden_layers $NUM_HIDDEN \
    --batch_size 64 \
    --num_epochs 1500 \
    --patience 1500 \
    --label_mode coarse \
    --target_mode framewise \
    --no_timestamp \
    --no_standardize \
    --optimizer adam

PYTHONPATH='.' python classifier/sonyc_ust/evaluate_predictions.py $OUTPUTDIR/output_mean.csv $ANNPATH $TAXPATH $OUTPUTDIR $GSHEET_ID $GOOGLE_DEV_APP_NAME
