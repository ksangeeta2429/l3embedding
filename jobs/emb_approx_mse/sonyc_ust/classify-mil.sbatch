#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=mil-classify-ust
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --output="mil-classify-ust-%j.out"
#SBATCH --err="mil-classify-ust-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`
module purge
module load cudnn/9.0v7.3.0.29
source activate l3embedding-tf-12-gpu

EPATH=$1
MODE=$2
BATCH_SIZE=$3
TIMESCALE=$4
EMBLEN=$5

SRCDIR=.
ANNPATH=/beegfs/dr2915/sonyc_ust/data/annotations.csv
TAXPATH=/beegfs/dr2915/sonyc_ust/data/dcase-ust-taxonomy.yaml
EMBDIR=/scratch/sk7898/embedding_approx_mse/features/sonyc_ust/pca/${MODE}/${TIMESCALE}/${BATCH_SIZE}/pca_batch_${BATCH_SIZE}_len_${EMBLEN}_kernel_linear/${EPATH}
OUTDIR=$SCRATCH/embedding_approx_mse/classifier/sonyc_ust/mil

if [ $# -gt 5 ]; then
    NUM_HIDDEN=$6
    HIDDEN_SIZE=$7
else
    NUM_HIDDEN=0
    HIDDEN_SIZE=0
fi

EXPDIR=$OUTDIR/${EPATH}/${NUM_HIDDEN}_${HIDDEN_SIZE}
OUTPUTDIR=$EXPDIR/results

echo "NUM_HIDDEN=$NUM_HIDDEN"
echo "HIDDEN_SIZE=$HIDDEN_SIZE"

GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo'

# Sample jobs: 
# sbatch classify-mil.sbatch 8000_64_160_1024_half_fmax_None/20200908180725 
# sbatch classify-mil.sbatch 8000_64_160_1024_half_fmax_None/20200908180725 1 128

python classifier/sonyc_ust/classify.py $ANNPATH $TAXPATH $EMBDIR $EXPDIR results \
    --hidden_layer_size $HIDDEN_SIZE \
    --num_hidden_layers $NUM_HIDDEN \
    --batch_size 64 \
    --num_epochs 1500 \
    --patience 1500 \
    --label_mode coarse \
    --target_mode mil \
    --no_timestamp \
    --optimizer adam

PYTHONPATH='.' python classifier/sonyc_ust/evaluate_predictions.py $OUTPUTDIR/output.csv $ANNPATH $TAXPATH $OUTPUTDIR $GSHEET_ID $GOOGLE_DEV_APP_NAME
