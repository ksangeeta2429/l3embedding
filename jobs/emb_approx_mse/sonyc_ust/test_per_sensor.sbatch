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

SRCDIR=.
ANNPATH=/scratch/sk7898/sonyc_ust/data/annotations.csv
TAXPATH=/scratch/sk7898/sonyc_ust/data/dcase-ust-taxonomy.yaml
OUTPUTDIR=$SCRATCH/sonyc_output/embeddings/classifier/sonyc_ust/mil/melSpec_20200304183233_48000_256_242_2048/0_0/results

echo "NUM_HIDDEN=$NUM_HIDDEN"
echo "HIDDEN_SIZE=$HIDDEN_SIZE"

GOOGLE_DEV_APP_NAME='l3compression'
GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo'

PYTHONPATH='.' python classifier/sonyc_ust/evaluate_predictions.py $OUTPUTDIR/output.csv $ANNPATH $TAXPATH $OUTPUTDIR $GSHEET_ID $GOOGLE_DEV_APP_NAME
