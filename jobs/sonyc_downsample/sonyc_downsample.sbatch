#!/usr/bin/env bash

#SBATCH --job-name=sonyc-downsample
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --output="sonyc-downsample-%j.out"
#SBATCH --err="sonyc-downsample-%j.err"

source ~/.bashrc
source activate l3embedding-cpu
cd `git rev-parse --show-toplevel`

module purge

SRCDIR=.

SAMPLE_SIZE=60028928
SONYC_CSV=/scratch/dr2915/reduced_embeddings/sonyc_files_list.csv
OUTPUT_DIR=$SCRATCH/sonyc_ndata=$SAMPLE_SIZE

echo SONYC_CSV=$SONYC_CSV
echo SAMPLE_SIZE=$SAMPLE_SIZE
echo OUTPUT_DIR=$OUTPUT_DIR

python -u $SRCDIR/01_downsample_sonyc_points.py \
  --random-state 20180216 \
  $SONYC_CSV \
  $OUTPUT_DIR \
  $SAMPLE_SIZE
