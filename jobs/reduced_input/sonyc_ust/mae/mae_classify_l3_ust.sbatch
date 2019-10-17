#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=classify-ust-loso
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=1-0
#SBATCH --mail-type=FAIL
#SBATCH --output="classify-ust-mil-%j.out"

source ~/.bashrc
cd `git rev-parse --show-toplevel`
source activate l3embedding-new-cpu

module purge
module load cudnn/9.0v7.3.0.29

SRCDIR=.
ANNPATH=/beegfs/jtc440/sonyc_ust/data/annotations.csv
TAXPATH=/beegfs/jtc440/sonyc_ust/data/dcase-ust-taxonomy.yaml
EMBDIR=$SCRATCH/embeddings/model_best_valid_mae/features/sonyc_ust/l3/embedding_approx_mse/$1
OUTDIR=$SCRATCH/embeddings/model_best_valid_mae/classifier/sonyc_ust/mil

EXPDIR=$OUTDIR/$1
OUTPUTDIR=$EXPDIR/results

GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo'

if [ $# -gt 1 ]; then
    NUM_HIDDEN=$2
    HIDDEN_SIZE=$3
else
    NUM_HIDDEN=0
    HIDDEN_SIZE=0
fi

echo "NUM_HIDDEN=$NUM_HIDDEN"
echo "HIDDEN_SIZE=$HIDDEN_SIZE"

#module purge
#module load sox/intel/14.4.2
#module load ffmpeg/intel/3.2.2
#module load rubberband/intel/1.8.1

python $SRCDIR/classifier/sonyc_ust/classify.py $ANNPATH $TAXPATH $EMBDIR $EXPDIR results \
    --hidden_layer_size $HIDDEN_SIZE \
    --num_hidden_layers $NUM_HIDDEN \
    --batch_size 64 \
    --num_epochs 1500 \
    --patience 1500 \
    --label_mode coarse \
    --target_mode mil \
    --no_timestamp \
    --optimizer adam

PYTHONPATH='.' python $SRCDIR/classifier/sonyc_ust/evaluate_predictions.py $OUTPUTDIR/output.csv $ANNPATH $TAXPATH $OUTPUTDIR $GSHEET_ID $GOOGLE_DEV_APP_NAME