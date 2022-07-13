#!/bin/bash

backbone_weights=ssl-pretrained
batch_size=1024
num_workers=32

prediction_type=classification
#prediction_type=regression

# data_path=/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/data/south/
#data_path=/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/data/morphology_labels_smooth-or-featured_smooth_fraction_test_15700.h5

# data_path=/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/data/lens_labels_val_20941.h5
data_path=/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/data/lens_labels_test_23267.h5

#checkpoint_name=ssl-pretrained_nlabels101650_epoch075_lr0.001_lrbb0.001
#checkpoint_name=imagenet_nlabels101650_lr0.001_lrbb0.001
# checkpoint_name=random_nlabels101650_lr0.001_lrbb0.001

# checkpoint_name=ssl-pretrained_nlabels188468_bs512_lr0.03_tau0.1_epoch099_lr0.01_lrbb0.0_epoch=03_val_Accuracy=0.9909=0.9924
# checkpoint_name=ssl-pretrained_nlabels188468_bs512_lr0.03_tau0.1_epoch099_lr0.001_lrbb0.001_epoch=97_val_Accuracy=0.9941

global_args="--data_path $data_path  
      --batch_size $batch_size	
      --prediction_type $prediction_type
      --num_workers $num_workers
      --overwrite
      --gpu
      -v"

# 
# for backbone in resnet18
# do
#     for epoch in 030 #000 010 040 075
#     do

# 	    #checkpoint_path=../trained_models/${backbone}/finetuning/ssl-pretrained_nlabels22500_epoch${epoch}_lr0.05_lrbb0.0.ckpt
# #	    OUTPUT_DIR=../trained_models/${backbone}/finetuning/predictions/ssl-pretrained_nlabels22500_epoch${epoch}_lr0.05_lrbb0.0/

# 	    checkpoint_path=../trained_models/${backbone}/finetuning/strong-lens/${checkpoint_name}.ckpt
# 	    OUTPUT_DIR=../trained_models/${backbone}/finetuning/strong-lens/predictions/${checkpoint_name}/
	    
# 	    args="$global_args --OUTPUT_DIR $OUTPUT_DIR --checkpoint_path $checkpoint_path"
# 	    echo $args
# 	    srun python predict.py $args
	    
#     done
# done

# Run on all files in directory
model_dir=../trained_models/resnet50_32node/finetuning/strong-lens/best_models/
for checkpoint_path in ${model_dir}*.ckpt
do

	# Extract filename from path
	model_name=`echo "$checkpoint_path" | sed -r "s/.+\/(.+)\..+/\1/"`
	OUTPUT_DIR=${model_dir}/predictions/test/${model_name}

	args="$global_args --OUTPUT_DIR $OUTPUT_DIR --checkpoint_path $checkpoint_path"

	echo $args
	python predict.py $args

done


# for backbone in resnet18
# do

#     for file_head in linear finetuned random
#     do

# 	for lab in train val
# 	do
# 	    checkpoint_path=../trained_models/${backbone}/finetuning/best_${file_head}.ckpt
# 	    OUTPUT_DIR=../trained_models/${backbone}/finetuning/predictions/
# 	    label_path=/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/data/morphology_labels_smooth-or-featured_smooth_fraction_${lab}.npy
	    
# 	    args="$global_args --OUTPUT_DIR $OUTPUT_DIR --checkpoint_path $checkpoint_path
# 	    --file_head ${file_head}_${lab}
# 	    --label_path $label_path"
# 	    echo $args
# 	    python predict.py $args
# 	done
#     done
# done

