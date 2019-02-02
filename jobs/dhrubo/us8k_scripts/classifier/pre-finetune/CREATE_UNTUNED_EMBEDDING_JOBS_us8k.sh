#!/usr/bin/env bash
features=( `find $SCRATCH/embeddings/features/us8k/l3comp/short/pruned_model/ -maxdepth 1 -mindepth 1 -type d` )

for f in ${features[@]}; do
	filename=`basename $f`
	outname=jobs_classifier_train_array_$filename.sh
	rm -f $outname
	for i in `seq 1 10`; do
		echo sbatch classifier-train-array-us8k.sbatch $f $i >> $outname
		echo sleep 1 >> $outname
	done
done
