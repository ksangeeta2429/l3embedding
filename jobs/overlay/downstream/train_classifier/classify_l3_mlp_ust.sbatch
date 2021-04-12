#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=classify-ust-mlp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output="classify-ust-mlp-%j.out"

source ~/.bashrc
cd `git rev-parse --show-toplevel`

module purge

export SRCDIR=.
export VERSION='v0.4'
export SONYC_UST_PATH=/scratch/work/sonyc/sonyc/ust
export TAXPATH=$SONYC_UST_PATH/annotations/$VERSION/dcase-ust-taxonomy.yaml
export EMBDIR=$SCRATCH/sonyc_output/embeddings/$VERSION/features/sonyc_ust/l3/$1
export OUTDIR=$SCRATCH/sonyc_output/embeddings/$VERSION/classifier/sonyc_ust/mlp

if [ "$VERSION" = "v2.2" ]; then
    export ANNPATH=$SONYC_UST_PATH/annotations/latest/annotations_w_test_anns.csv
else
    export ANNPATH=$SONYC_UST_PATH/annotations/$VERSION/annotations.csv
fi

if [ $# -gt 1 ]; then
    export NUM_HIDDEN=$2
    export HIDDEN_SIZE=$3
else
    export NUM_HIDDEN=0
    export HIDDEN_SIZE=0
fi

export EXPDIR=$OUTDIR/$1/${NUM_HIDDEN}_${HIDDEN_SIZE}
export OUTPUTDIR=$EXPDIR/results

echo "SONYC_VERSION=$VERSION"
echo "NUM_HIDDEN=$NUM_HIDDEN"
echo "HIDDEN_SIZE=$HIDDEN_SIZE"

export GOOGLE_DEV_APP_NAME='l3compression'
export GSHEET_ID='1iNfqw0mvBDSSvuFkj5nfu_Wg6lj7i6uZ59a2YJ694fo'

# sbatch classify_l3_mlp_ust.sbatch melSpec_20200304183233_48000_256_242_2048
# sbatch classify_l3_mlp_ust.sbatch melSpec_20200304183233_48000_256_242_2048 1 128
# sbatch classify_l3_mlp_ust.sbatch melSpec_20200304183233_48000_256_242_2048 2 128
# sbatch classify_l3_mlp_ust.sbatch melSpec_20200304183233_48000_256_242_2048 1 256
# sbatch classify_l3_mlp_ust.sbatch melSpec_20200304183233_48000_256_242_2048 2 256
singularity exec --nv \
	    --overlay /scratch/sk7898/overlay/tf2.2-cuda10.1.ext3:ro \
	    /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif \
	    bash -c "source /ext3/env.sh;
        python /scratch/sk7898/l3embedding/classifier/sonyc_ust/classify.py $ANNPATH $TAXPATH $EMBDIR $EXPDIR results \
            --hidden_layer_size $HIDDEN_SIZE \
            --num_hidden_layers $NUM_HIDDEN \
            --batch_size 64 \
            --num_epochs 1500 \
            --patience 20 \
            --label_mode coarse \
            --target_mode framewise \
            --no_timestamp \
            --optimizer adam;
        PYTHONPATH='.' python classifier/sonyc_ust/evaluate_predictions.py $OUTPUTDIR/output_mean.csv $ANNPATH $TAXPATH $OUTPUTDIR $GSHEET_ID $GOOGLE_DEV_APP_NAME"
