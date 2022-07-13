#!/bin/bash 
#SBATCH -C gpu
#SBATCH -A m3900
#SBATCH --qos=early_science #regular
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task 32
#SBATCH --time=12:00:00
#SBATCH -o sout/representations_%x-%j.out                               

main_dir=/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/ssl-legacysurvey-pl/scripts

cd $main_dir

module load python
conda activate ssl-pl

backbone_weights=ssl-pretrained
batch_size=1024
num_workers=16

prediction_type=classification
#prediction_type=regression

data_path=/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/data/south/

global_args="--data_path $data_path  
      --batch_size $batch_size	
      --num_workers $num_workers
      --extract_representations 
      --gpu
      -v"

for backbone in resnet50
do

    checkpoint_path=../trained_models/${backbone}/bs256_lr0.24_tau0.2_epoch=1499.ckpt
    output_dir=../trained_models/${backbone}/representations/

    args="$global_args --output_dir $output_dir --checkpoint_path $checkpoint_path --file_head representations --extract_representations" 
    #echo $args
    
    # Linear classification
    srun python predict.py $args

done

