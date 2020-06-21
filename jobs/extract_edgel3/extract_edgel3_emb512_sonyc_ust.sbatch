#!/bin/bash

#SBATCH --job-name=sonyc-ust-extract-edgel3-emb512
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --output="sonyc-ust-ext-edgel3-emb512-%j.out"

source ~/.bashrc
source activate edgel3

module purge
module load cuda/10.1.105
module load sox/intel/14.4.2      
module load ffmpeg/intel/3.2.2    
module load rubberband/intel/1.8.1

SRCDIR=`git rev-parse --show-toplevel`/classifier/sonyc_ust

cd $SRCDIR
DATAPATH=/beegfs/dr2915/sonyc_ust/data
ANNPATH=$DATAPATH/annotations.csv 
TAXPATH=$DATAPATH/dcase-ust-taxonomy.yaml 
EMBDIR=$SCRATCH/embeddings/features/sonyc_ust/l3/edgel3

echo "DATA_DIR=$DATAPATH"
echo "EMB_DIR=$EMBDIR"

python extract_embedding.py $ANNPATH $DATAPATH $EMBDIR edgel3 \
    --retrain_type $1 \
    --sparsity 95.45 \
    --hop_duration 0.1 \
    --progress
