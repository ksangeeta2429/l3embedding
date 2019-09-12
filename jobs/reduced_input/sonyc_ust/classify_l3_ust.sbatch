#!/usr/bin/env bash

##SBATCH --gres=gpu:1
##SBATCH --job-name=classify-ust-loso
##SBATCH --nodes=1
##SBATCH --cpus-per-task=1
##SBATCH --mem=20GB
##SBATCH --time=1-0
##SBATCH --mail-type=FAIL
#SBATCH --output="classify-ust-loso-%j.out"

source ~/.bashrc
cd `git rev-parse --show-toplevel`
source activate l3embedding-new-cpu

module purge
module load cudnn/9.0v7.3.0.29

SRCDIR=.
ANNPATH=/beegfs/jtc440/urban_sound_tagging_final/annotations.csv
TAXPATH=/beegfs/jtc440/urban_sound_tagging_final/dcase-ust-taxonomy.yaml
EMBDIR=$SCRATCH/embeddings/features/sonyc_ust/l3/reduced_input/$1
OUTDIR=$SCRATCH/sensor_embedding_output/downstream_models/loso

EXPDIR=$OUTDIR/$1
OUTPUTDIR=$EXPDIR/results

#module purge
#module load sox/intel/14.4.2
#module load ffmpeg/intel/3.2.2
#module load rubberband/intel/1.8.1

cd $SRCDIR/classifier/sonyc_ust

python classify_loso.py $ANNPATH $TAXPATH $EMBDIR $EXPDIR results \
    --hidden_layer_size 0 \
    --num_hidden_layers 0 \
    --batch_size 64 \
    --num_epochs 1500 \
    --patience 1500 \
    --label_mode coarse \
    --target_mode mil \
    --no_timestamp \

python evaluate_loso.py $OUTPUTDIR $ANNPATH $TAXPATH