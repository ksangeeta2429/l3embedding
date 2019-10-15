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

ANNPATH=/beegfs/jtc440/sonyc_ust/data/annotations.csv
TAXPATH=/beegfs/jtc440/sonyc_ust/data/dcase-ust-taxonomy.yaml
EMBDIR=$SCRATCH/reduced_embeddings/umap/$1/$2/sonyc_ust/l3-mel256-music-512
OUTDIR=$SCRATCH/embeddings/umap_direct/classifier/sonyc_ust/l3-mel256-music-512/mil

EXPDIR=$OUTDIR/$1/$2
OUTPUTDIR=$EXPDIR/results

echo "Embedding directory: $EMBDIR"
echo "Output directory: $OUTPUTDIR"

GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo'

#module purge
#module load sox/intel/14.4.2
#module load ffmpeg/intel/3.2.2
#module load rubberband/intel/1.8.1

python classifier/sonyc_ust/classify.py $ANNPATH $TAXPATH $EMBDIR $EXPDIR results \
    --hidden_layer_size 0 \
    --num_hidden_layers 0 \
    --batch_size 64 \
    --num_epochs 1500 \
    --patience 1500 \
    --label_mode coarse \
    --target_mode mil \
    --no_timestamp \
    --optimizer adam

PYTHONPATH='.' python classifier/sonyc_ust/evaluate_predictions.py $OUTPUTDIR/output.csv $ANNPATH $TAXPATH $OUTPUTDIR $GSHEET_ID $GOOGLE_DEV_APP_NAME