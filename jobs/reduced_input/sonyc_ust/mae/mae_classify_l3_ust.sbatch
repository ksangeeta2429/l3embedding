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

#module purge
#module load sox/intel/14.4.2
#module load ffmpeg/intel/3.2.2
#module load rubberband/intel/1.8.1

cd $SRCDIR/classifier/sonyc_ust

python classify.py $ANNPATH $TAXPATH $EMBDIR $EXPDIR results \
    --hidden_layer_size 0 \
    --num_hidden_layers 0 \
    --batch_size 64 \
    --num_epochs 1500 \
    --patience 1500 \
    --label_mode coarse \
    --target_mode mil \
    --no_timestamp \
    --optimizer adam

python evaluate_predictions.py $OUTPUTDIR/output.csv $ANNPATH $TAXPATH $OUTPUTDIR