features_dir=$1
model=`basename $features_dir`
#model="$(cut -d'/' -f13 <<<"$features_dir")"
asr="$(cut -d'_' -f1 <<<"$model")" 

for f in $features_dir/*; do
    model_id=`basename $f`
    #echo $model_id
    outname=jobs_sonyc_${asr}_classifier_train_${model_id}.sh
    rm -f $outname
    for i in `seq 1 10`; do
	echo sbatch classifier-train-array-us8k.sbatch $f $i >> $outname
	echo sleep 5 >> $outname
    done
done
