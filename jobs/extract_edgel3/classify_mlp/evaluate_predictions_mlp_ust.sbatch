#!/usr/bin/env bash

#SBATCH --job-name=eval-edgel3-ust-mlp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=1-0
#SBATCH --mail-type=FAIL
#SBATCH --output="eval-edgel3-ust-mlp-%j.out"
#SBATCH --err="eval-edgel3-ust-mlp-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge
module load cudnn/9.0v7.3.0.29

if [ $USER == "sk7898" ]; then
    source activate l3embedding-tf-12-gpu
else
    source activate l3embedding-new-cpu
fi

SRCDIR=.
RETRAIN=$1
ANNPATH=/beegfs/dr2915/sonyc_ust/data/annotations.csv
TAXPATH=/beegfs/dr2915/sonyc_ust/data/dcase-ust-taxonomy.yaml
OUTDIR=$SCRATCH/embeddings/classifier/sonyc_ust/mlp
NUM_HIDDEN=$2 #2
HIDDEN_SIZE=$3 #128

EXPDIR=$OUTDIR/edgel3/edgel3-${RETRAIN}-95.45-512/${NUM_HIDDEN}_${HIDDEN_SIZE}
OUTPUTDIR=$EXPDIR/results

GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo'

PYTHONPATH='.' python classifier/sonyc_ust/evaluate_predictions.py $OUTPUTDIR/output_mean.csv $ANNPATH $TAXPATH $OUTPUTDIR $GSHEET_ID $GOOGLE_DEV_APP_NAME
