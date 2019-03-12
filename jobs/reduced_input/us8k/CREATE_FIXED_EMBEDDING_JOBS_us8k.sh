#!/usr/bin/env bash

#features=( `find /scratch/dr2915/embeddings/features/us8k/l3 -maxdepth 1 -mindepth 1 -type d` )
features=('/scratch/dr2915/embeddings/features/us8k/l3/16k_64_50/music/cnn_L3_melspec2')

for f in ${features[@]}; do
	#filename=`basename $f`
	#outname=jobs_classifier_train_array_fixed_$filename.sh
	outname=jobs_classifier_train_array_16k_64_50.sh
	rm -f $outname
	for i in `seq 1 10`; do
		echo sbatch classifier-train-array-us8k.sbatch $f $i >> $outname
		echo sleep 1 >> $outname
	done
done

