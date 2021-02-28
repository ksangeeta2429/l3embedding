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

export SRCDIR=.
export VERSION='v0.4'
export CLS=$2
export SONYC_UST_PATH=/scratch/work/sonyc/sonyc/ust
export ANNPATH=$SRCDIR/notebooks/data/$VERSION/$CLS/annotations.csv
#export ANNPATH=$SONYC_UST_PATH/annotations/$VERSION/annotations.csv
export TAXPATH=$SONYC_UST_PATH/annotations/$VERSION/dcase-ust-taxonomy.yaml
export EMBDIR=$SCRATCH/sonyc_output/embeddings/$VERSION/features/sonyc_ust/l3/$1
export OUTDIR=$SCRATCH/sonyc_output/embeddings/$VERSION/classifier/sonyc_ust/mil/binary

if [ $# -gt 2 ]; then
    export NUM_HIDDEN=$3
    export HIDDEN_SIZE=$4
else
    export NUM_HIDDEN=0
    export HIDDEN_SIZE=0
fi

export EXPDIR=$OUTDIR/$1/$CLS/${NUM_HIDDEN}_${HIDDEN_SIZE}
export OUTPUTDIR=$EXPDIR/results


echo "CLASS=$CLS"
echo "NUM_HIDDEN=$NUM_HIDDEN"
echo "HIDDEN_SIZE=$HIDDEN_SIZE"


# Sample jobs: 
# sbatch classify-mil-binary.sbatch melSpec_20200304183233_48000_256_242_2048 engine 
# sbatch classify-mil-binary.sbatch melSpec_20200304183233_48000_256_242_2048 engine 1 128

singularity exec --nv \
	    --overlay /scratch/sk7898/overlay/tf2.2-cuda10.1.ext3:ro \
	    /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif \
	    bash -c "source /ext3/env.sh;
        python classifier/sonyc_ust/classify_binary.py $ANNPATH $TAXPATH $EMBDIR $EXPDIR results $CLS \
            --hidden_layer_size $HIDDEN_SIZE \
            --num_hidden_layers $NUM_HIDDEN \
            --batch_size 32 \
            --num_epochs 1500 \
            --patience 10 \
            --label_mode coarse \
            --target_mode mil \
            --out_all \
            --no_timestamp \
            --optimizer adam"

