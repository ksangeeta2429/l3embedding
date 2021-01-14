#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=classify-ust
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=1-0
#SBATCH --mail-type=FAIL
#SBATCH --output="classify-ust-mil-%j.out"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge

export SRCDIR=.
export VERSION='v0.4'
export SONYC_UST_PATH=/scratch/work/sonyc/sonyc/ust
export ANNPATH=$SONYC_UST_PATH/annotations/$VERSION/annotations.csv
export TAXPATH=$SONYC_UST_PATH/annotations/$VERSION/dcase-ust-taxonomy.yaml
export EMBDIR=$SCRATCH/sonyc_output/embeddings/$VERSION/features/sonyc_ust/l3/reduced_input/$1
#export EMBDIR=$SCRATCH/sonyc_output/embeddings/$VERSION/features/sonyc_ust/l3/reduced_input/$1
export OUTDIR=$SCRATCH/sonyc_output/embeddings/$VERSION/classifier/sonyc_ust/mil

if [ $# -gt 1 ]; then
    export NUM_HIDDEN=$2
    export HIDDEN_SIZE=$3
else
    export NUM_HIDDEN=0
    export HIDDEN_SIZE=0
fi

export EXPDIR=$OUTDIR/$1/${NUM_HIDDEN}_${HIDDEN_SIZE}
export OUTPUTDIR=$EXPDIR/results

echo "NUM_HIDDEN=$NUM_HIDDEN"
echo "HIDDEN_SIZE=$HIDDEN_SIZE"

export GOOGLE_DEV_APP_NAME='l3compression'
export GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo'

singularity exec --nv \
	    --overlay /scratch/sk7898/overlay/tf2.2-cuda10.1.ext3:ro \
	    /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif \
	    bash -c "source /ext3/env.sh;
        python /scratch/sk7898/l3embedding/classifier/sonyc_ust/classify.py $ANNPATH $TAXPATH $EMBDIR $EXPDIR results \
            --hidden_layer_size $HIDDEN_SIZE \
            --num_hidden_layers $NUM_HIDDEN \
            --batch_size 64 \
            --num_epochs 1500 \
            --patience 1500 \
            --label_mode coarse \
            --target_mode mil \
            --no_timestamp \
            --optimizer adam;
        PYTHONPATH='.' python classifier/sonyc_ust/evaluate_predictions.py $OUTPUTDIR/output.csv $ANNPATH $TAXPATH $OUTPUTDIR $GSHEET_ID $GOOGLE_DEV_APP_NAME"
