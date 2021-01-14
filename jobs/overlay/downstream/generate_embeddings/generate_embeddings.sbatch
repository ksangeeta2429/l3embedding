#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=generate-embs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --output="generate-embs-%j.out"
#SBATCH --err="generate-embs-%j.err"

module purge

cd `git rev-parse --show-toplevel`

export VERSION='v0.4'
export SAMPLING_RATE=8000 #48000
export NUM_MELS=64 #256
export HOP_LEN=160 #242
export DFT=1024 #2048
#export EMBS_DIR=$SCRATCH/sonyc_output/models/openl3/environmental/audio_models
export L3_MODEL_PATH=$SCRATCH/sonyc_output/models/reduced_input/embedding/environmental/audio_models/l3_audio_melSpec_20201015214759_8000_64_160_1024.h5
#export L3_MODEL_PATH=$EMBS_DIR/l3_audio_melSpec_20200304183233_48000_256_242_2048.h5
export SONYC_UST_PATH=/scratch/work/sonyc/sonyc/ust
export DATA_DIR=$SONYC_UST_PATH/audio/48k
export METADATA_PATH=$SONYC_UST_PATH/annotations/$VERSION/annotations.csv
export OUTPUT_DIR=$SCRATCH/sonyc_output/embeddings/$VERSION
export DATASET='sonyc_ust'

singularity exec --nv \
	    --overlay /scratch/sk7898/overlay/tf2.2-cuda10.1.ext3:ro \
	    /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif \
	    bash -c "source /ext3/env.sh;         
                python /scratch/sk7898/l3embedding/05_generate_embedding_samples.py \
                        --random-state 20180302 \
                        --verbose \
                        --features 'l3' \
                        --l3embedding-model-path $L3_MODEL_PATH \
                        --hop-size 0.1 \
                        --gpus 0 \
                        --samp-rate $SAMPLING_RATE \
                        --num-mels $NUM_MELS \
                        --mel-hop-length $HOP_LEN \
                        --num-dft $DFT \
                        --sonyc-ust-annotation-path $METADATA_PATH \
                        --with-melSpec \
                        $DATASET \
                        $DATA_DIR \
                        $OUTPUT_DIR"
