features=( `find /scratch/sk7898/embeddings/features/esc50/l3comp/short/fixed -maxdepth 1 -mindepth 1 -type d` )

for f in ${features[@]}; do
	filename=`basename $f`
	outname=jobs_classifier_train_array_fixed_$filename.sh
	rm -f $outname
	for i in `seq 1 5`; do
		echo sbatch classifier-train-array-esc50.sbatch $f $i >> $outname
		echo sleep 1 >> $outname
	done
done

