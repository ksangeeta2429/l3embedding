#!/usr/bin/env bash
features_dir="$SCRATCH/embeddings/features/us8k/l3/reduced_input"

for f in $features_dir/*; do
    #folder=${f}/fixed
    model_id=`basename $f`
    #echo $folder
    #echo $model_id
    outname=jobs_baseline_student_classifier_train_$model_id.sh
    rm -f $outname
    for i in `seq 1 10`; do
	    echo sbatch classifier-train-array-us8k.sbatch $f $i >> $outname
	    if [ $i -lt 10 ]; then
	        echo sleep 30 >> $outname
	    fi
    done
done

