#features_dir='/scratch/sk7898/embedding_approx_mse/features/esc50/pca/dpp/day/500000/pca_batch_500000_len_128_kernel_linear/8000_64_160_1024_half_fmax_None'

features_dir='/scratch/sk7898/embedding_approx_mse/features/esc50/mse_original/8000_64_160_1024_fmax_None'

for f in $features_dir/*; do
    folder=${f}
    model_id=`basename $f`
    #echo $folder
    #echo $model_id
    outname=jobs_classifier_train_$model_id.sh
    rm -f $outname
    for i in `seq 1 5`; do
	echo sbatch classifier-train-array-esc50.sbatch $folder $i >> $outname
	echo sleep 1 >> $outname
    done
done

