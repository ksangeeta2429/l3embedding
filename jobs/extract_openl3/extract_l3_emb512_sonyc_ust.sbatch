#!/bin/bash

#SBATCH --job-name=sonyc-ust-extract_l3_emb512
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --output="sonyc-ust-ext-l3-emb512-%j.out"

source ~/.bashrc
source activate ust

module purge
module load sox/intel/14.4.2      
module load ffmpeg/intel/3.2.2    
module load rubberband/intel/1.8.1

SRCDIR=`git rev-parse --show-toplevel`/classifier/sonyc_ust

cd $SRCDIR
DATAPATH=/beegfs/dr2915/sonyc_ust/data
AUDIOPATH=$DATAPATH/audio
ANNPATH=$DATAPATH/annotations.csv 
TAXPATH=$DATAPATH/dcase-ust-taxonomy.yaml 
EMBDIR=$SCRATCH/embeddings/features/sonyc_ust/l3/openl3

python extract_embedding.py $ANNPATH $AUDIOPATH $EMBDIR l3 \
    --l3_input_repr mel256 \
    --l3_content_type music \
    --l3_embedding_size 512 \
    --hop_duration 0.1 \
    --progress
