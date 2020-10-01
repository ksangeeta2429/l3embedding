#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=classify-ust-mlp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --output="classify-ust-mlp-%j.out"
#SBATCH --err="classify-ust-mlp-%j.err"

source ~/.bashrc
cd `git rev-parse --show-toplevel`
module purge
module load cudnn/9.0v7.3.0.29
source activate l3embedding-tf-12-gpu

SRCDIR=.
ANNPATH=/beegfs/dr2915/sonyc_ust/data/annotations.csv
TAXPATH=/beegfs/dr2915/sonyc_ust/data/dcase-ust-taxonomy.yaml
EMBDIR=$SCRATCH/embedding_approx_mse/features/sonyc_ust/pca/dpp/day/500000/pca_batch_500000_len_256_kernel_linear/$1
OUTDIR=$SCRATCH/embedding_approx_mse/classifier/sonyc_ust/mlp

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

# Sample jobs: 
# sbatch classify-mlp.sbatch 8000_64_160_1024_half_fmax_None/20200908180725 
# sbatch classify-mlp.sbatch 8000_64_160_1024_half_fmax_None/20200908180725 1 128

python classifier/sonyc_ust/classify.py $ANNPATH $TAXPATH $EMBDIR $EXPDIR results \
    --hidden_layer_size $HIDDEN_SIZE \
    --num_hidden_layers $NUM_HIDDEN \
    --batch_size 64 \
    --num_epochs 1500 \
    --patience 1500 \
    --label_mode coarse \
    --target_mode framewise \
    --no_timestamp \
    --optimizer adam

PYTHONPATH='.' python classifier/sonyc_ust/evaluate_predictions.py $OUTPUTDIR/output_mean.csv $ANNPATH $TAXPATH $OUTPUTDIR $GSHEET_ID $GOOGLE_DEV_APP_NAME
