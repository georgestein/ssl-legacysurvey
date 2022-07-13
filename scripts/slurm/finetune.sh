#!/bin/bash 
#SBATCH -C gpu
#SBATCH -A m3900
#SBATCH --qos=early_science #regular
#SBATCH --nodes 1
#SBATCH --gpus 4
#SBATCH --cpus-per-task 32
#SBATCH --time=12:00:00
#SBATCH -o sout/finetune_lens_grrrssgbjcgnr_%x-%j.out                               

module load python
conda activate ssl-pl
export HDF5_USE_FILE_LOCKING=FALSE

main_dir=/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/ssl-legacysurvey-pl/scripts
cd $main_dir

batch_size=256
num_workers=32

prediction_type=classification
#prediction_type=regression

learning_rate_head_linear=0.01 
learning_rate_head_finetune=0.001 
learning_rate_head_random=0.001
learning_rate_backbone=0.001
learning_rate_backbone_ssl=0.0001

max_epochs=100
#nlabels=1800
#nlabels=22500
#nlabels=24619
nlabels=188468

checkpoint_head=bs256_lr0.24_tau0.2

augmentations=grrrssgbjcgnrg 

# Morphology
#data_path=/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/data/morphology_labels_smooth-or-featured_smooth_fraction_train_${nlabels}.h5
#val_data_path=/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/data/morphology_labels_smooth-or-featured_smooth_fraction_val_2500.h5

# checkpoint_head=bs512_lr0.5_tau0.2
# checkpoint_head=bs512_lr0.03_tau0.1

# Lensing
data_path=/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/data/lens_labels_train_${nlabels}.h5
# val_data_path=/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/data/lens_labels_val_2736.h5
val_data_path=/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/data/lens_labels_val_20941.h5


for max_num_samples in 2048 #512 1024 2048 #4096 8192 16384 32768 65536 131072 $nlabels # Subsample the full training set size
do
	global_args="--data_path $data_path  
      --val_data_path $val_data_path
      --label_name labels  
      --batch_size $batch_size	
      --num_workers $num_workers
      --prediction_type $prediction_type
      --max_epochs $max_epochs
      --augmentations $augmentations
      --max_num_samples $max_num_samples
      --gpu
      -v"

	for backbone in resnet50 # for each backbone
	do
	    for epoch in 1499 
	    do
			

		checkpoint_path=../trained_models/${backbone}/${checkpoint_head}_epoch=${epoch}.ckpt

		output_dir=../trained_models/${backbone}/finetuning/strong-lens/
		output_file_tail=_${checkpoint_head}_epoch${epoch}

		local_args="--output_dir ${output_dir}/${checkpoint_head}_epoch=${epoch}/ntrain_${max_num_samples}/
			--output_file_tail $output_file_tail
			--checkpoint_path $checkpoint_path 
			--learning_rate_backbone $learning_rate_backbone_ssl 
			--backbone_weights ssl-pretrained
	        "

		# Linear classification
		#srun python finetune.py $global_args $local_args --learning_rate $learning_rate_head_linear 

		# Finetune whole model
		srun python finetune.py $global_args $local_args --learning_rate $learning_rate_head_finetune --finetune

	  	# Train from scratch
	  	output_file_tail=_
		local_args="--output_dir ${output_dir}/random/ntrain_${max_num_samples}/
		--output_file_tail $output_file_tail
		--learning_rate $learning_rate_head_random
		--learning_rate_backbone $learning_rate_backbone
		--backbone $backbone
		"
		# --resume_training
		# --checkpoint_path $checkpoint_path 
	 	#    "	

		#python finetune.py $global_args $local_args --backbone_weights random  
		srun python finetune.py $global_args $local_args --backbone_weights random --finetune
	  

		# Train from imagenet pretrained
		local_args="--output_dir ${output_dir}/imagenet/ntrain_${max_num_samples}/ 
		--output_file_tail $output_file_tail
		--learning_rate $learning_rate_head_random
		--learning_rate_backbone $learning_rate_backbone 
		--backbone $backbone
	    "	
	    
		#python finetune.py $global_args $local_args --backbone_weights imagenet  
		srun python finetune.py $global_args $local_args --backbone_weights imagenet --finetune 

	    done	
	done
done

