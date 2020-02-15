#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=classify-ust-mlp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=1-0
#SBATCH --mail-type=FAIL
#SBATCH --output="classify-ust-mlp-%j.out"

source ~/.bashrc
cd `git rev-parse --show-toplevel`
source activate l3embedding-new-cpu

module purge
module load cudnn/9.0v7.3.0.29

SRCDIR=.
ANNPATH=/beegfs/dr2915/sonyc_ust/annotations.csv
TAXPATH=/beegfs/dr2915/sonyc_ust/dcase-ust-taxonomy.yaml
#EMBDIR=/scratch/dr2915/embeddings/db_mel/features/sonyc_ust/l3/20191108201753_8000_64_160_1024_half_for_quantization_flattenOp
OUTDIR=$SCRATCH/embeddings/db_mel_classifier/sonyc_ust/mlp

#module purge
#module load sox/intel/14.4.2
#module load ffmpeg/intel/3.2.2
#module load rubberband/intel/1.8.1

#if [ $# -gt 1 ]; then
NUM_HIDDEN=2 #$1
HIDDEN_SIZE=128 #$2
#else
#    NUM_HIDDEN=0
#    HIDDEN_SIZE=0
#fi

#EXPDIR=$OUTDIR/20191108201753_8000_64_160_1024_half_for_quantization_flattenOp/${NUM_HIDDEN}_${HIDDEN_SIZE}
OUTPUTDIR=/scratch/dr2915/pi_ml_container/output/sonyc_ust/cmsis_test

echo "NUM_HIDDEN=$NUM_HIDDEN"
echo "HIDDEN_SIZE=$HIDDEN_SIZE"

GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo'

PYTHONPATH='.' python classifier/sonyc_ust/evaluate_predictions.py $OUTPUTDIR/output_mean.csv $ANNPATH $TAXPATH $OUTPUTDIR $GSHEET_ID $GOOGLE_DEV_APP_NAME
