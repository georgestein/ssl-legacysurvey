#!/bin/bash 
#SBATCH -C gpu
#SBATCH -A m3900
#SBATCH --qos=early_science #regular
#SBATCH --nodes 32
#SBATCH --gpus 128
#SBATCH --cpus-per-task 32
#SBATCH --time=12:00:00
#SBATCH -o sout/decals_grrrssgbjcgnr_%x-%j.out                               

main_dir=/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/ssl-legacysurvey-pl/scripts
moco_loc=train_ssl.py

cd $main_dir

train_file=/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/data/images_npix152_000000000_003500000.h5

module load python
conda activate ssl-pl

export HDF5_USE_FILE_LOCKING=FALSE

backbone=resnet18
#backbone=resnet34

#backbone=resnet50
num_nodes=32
batch_size=256
learning_rate=0.24
encoder_momentum=0.996
softmax_temperature=0.2
max_epochs=1500
num_workers=32
num_negatives=65536 

augmentations=grrrssgbjcgnrg

# To continue training from checkpoint
# ckpt_path=../trained_models/resnet50_32node/bs256_lr0.24_tau0.1_epoch=751.ckpt

date
#cp $train_file /tmp/train.h5 
#train_file=/tmp/train.h5 
args="--backbone $backbone --gpu --use_mlp --emb_dim 128
     --augmentations $augmentations
     --batch_size $batch_size --strategy ddp --num_workers $num_workers
     --num_nodes $num_nodes --learning_rate $learning_rate
     --num_negatives $num_negatives --data_path ${train_file}
     --max_epochs $max_epochs 
     --encoder_momentum $encoder_momentum --softmax_temperature $softmax_temperature
     --output_dir ../trained_models/${backbone}
     "
#     --ckpt_path $ckpt_path"    

srun python $moco_loc ${args}


