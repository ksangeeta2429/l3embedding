features_dir='/scratch/sk7898/embeddings/features/us8k/l3'

for f in $features_dir/*; do
    folder=${f}/fixed
    model_id=`basename $f`
    #echo $folder
    #echo $model_id
    outname=jobs_classifier_train_$model_id.sh
    rm -f $outname
    for i in `seq 1 10`; do
	echo sbatch classifier-train-array-us8k.sbatch $folder $i >> $outname
	echo sleep 1 >> $outname
    done
done

