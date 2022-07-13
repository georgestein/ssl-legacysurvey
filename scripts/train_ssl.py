"""Trains an encoder network using self-supervised learning. 

Usage example: 
> python train_ssl.py --data_path ../data/images.h5 --augmentations jcrg --output_dir ../experiments/test/
                    --batch_size 256 --learning_rate 0.03 --max_epochs 1 --max_num_samples 2048 

Important parameters for training


Currently supports MocoV2.

DDP training encouraged!
"""

# To prevent OpenBLAS blas_thread_init
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import h5py

import argparse
import logging

from pathlib import Path
import os
import sys
import glob

from pytorch_lightning import loggers as pl_loggers
# from pl_bolts.models.self_supervised import Moco_v2
from pytorch_lightning.plugins import DDPPlugin

from ssl_legacysurvey.moco.moco2_module import Moco_v2
from ssl_legacysurvey.data_loaders import datamodules
from ssl_legacysurvey.utils import format_logger

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='runtime parameters')
    # Data loading
    parser.add_argument("--data_path", type=str, default='/pscratch/sd/g/gstein/machine_learning/decals_self_supervised/data/images_npix152_000000000_003500000.h5',
                        help="Path to hdf5 data file")
    
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers for data loader")

    # Training 
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU")

    parser.add_argument("--gpus", type=int, default=-1,
                        help="Number of gpus to use")

    parser.add_argument("--num_nodes", type=int, default=1,
                        help="Number of gpu nodes available")

    # ddp does not work in ipython notebook, only ddp_spawn does
    parser.add_argument("--strategy", type=str, default='ddp', #'ddp_spawn',
                        help="Distributed training strategy")

    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Run only a few batches")

    parser.add_argument("--check_val_every_n_epoch", type=int, default=999,
                        help="How often to run validation epoch")
        
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=1,
                        help="Checkpoint model every n epochs")

    parser.add_argument("--save_top_k", type=int, default=5,
                        help="Top number of checkpoints to save")

    parser.add_argument("--num_sanity_val_steps", type=int, default=0,
                        help="Number of validation steps to run right after model initialization")

    parser.add_argument("--max_num_samples", type=int, default=None,
                        help="Maximum number of data samples to use. Defaults to full dataset size")

    # Augmentations
    parser.add_argument("--augmentations", type=str, default='grrrssgbjcgnrg',
                        help="2 character abbreviations of training augmentations to use")

    parser.add_argument("--val_augmentations", type=str, default='ccrg',
                        help="2 character abbreviations of validation augmentations to use")

    parser.add_argument("--jitter_lim", type=int, default=7,
                        help="Number of pixels in x,y to jitter image. (-jitter_lim, jitter_lim)")

    parser.add_argument("--only_dered", action="store_true",
                        help="Deredden if calling gr augmentation")

    parser.add_argument("--only_red", action="store_true",
                        help="Redden if calling gr augmentation")

    parser.add_argument("--ebv_max", type=float, default=1.0,
                        help="Maximum extinction reddening to use if calling gr augmentation")

    parser.add_argument("--gn_uniform", action="store_false",
                        help="Draw from uniform Gaussian noise if using gn augmentation")

    parser.add_argument("--gb_uniform", action="store_false",
                        help="Draw from uniform Gaussian blur if using gb augmentation")

    parser.add_argument("--gr_uniform", action="store_false",
                        help="Draw from uniform Galactic reddenning if using gr augmentation")

    # Optimizers
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for model training")

    parser.add_argument("--learning_rate", type=float, default=0.03,
                        help="Learning rate for model optimization")

    parser.add_argument("--encoder_momentum", type=float, default=0.996,
                        help="Mocov2 encoder momentum")

    parser.add_argument("--softmax_temperature", type=float, default=0.2,
                        help="Mocov2 softmax temperature")

    parser.add_argument("--max_epochs", type=int, default=1,
                        help="Max number of training epochs")

    parser.add_argument("--optimizer", type=str, default='SGD',
                        help="Optimizer to use - Mocov2 only accepts SGD", choices=['SGD'])

    parser.add_argument("--test_run", action="store_true",
                        help="Subsample training and validation data")

    parser.add_argument("--seed", type=int , default=13579,
                        help="random seed for train test split")

    # Model architecture and settings
    parser.add_argument("--backbone", type=str, default='resnet18',
                        help="Encoder architecture to use", choices=['resnet18', 'resnet34', 'resnet50', 'resnet152'])
    
    parser.add_argument("--use_mlp", action="store_true",
                        help="use projection head")

    parser.add_argument("--emb_dim", type=int, default=128,
                        help="Dimensionality where loss is calculated")

    parser.add_argument("--num_negatives", type=int, default=65536,
                        help="Number of negative samples to keep in queue")

    # Setup outputs and others
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Continue training from checkpoint on disk")

    parser.add_argument("--output_dir", type=str, default='../experiments/test/',
                        help="directory to save trained model and logs")

    parser.add_argument("--logfile_name", type=str, default='ssl_train.log',
                        help="name of log file")
        
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")

    args = parser.parse_args()

    return args

def main(args):
    """Sets up model, sets training parameters, configures training callbacks, then trains model"""

    params = vars(args) # convert args to dictionary
    params['ssl_training'] = True
    
    pl.seed_everything(params['seed'], workers=True)
    Path(params['output_dir']).mkdir(parents=True, exist_ok=True)

    logger = format_logger.create_logger(
        filename=os.path.join(params['output_dir'], params['logfile_name']),
        )

    logger.info("\nTraining with the following parameters:")
    for k, v in params.items():
        logger.info(f"{k}: {v}")

    file_output_head = f"bs{params['batch_size']}_lr{params['learning_rate']}_tau{params['softmax_temperature']}"

    if params['ckpt_path']:
        # Load pretrained backbone from checkpoint on disk
        model = Moco_v2.load_from_checkpoint(
            checkpoint_path=params['ckpt_path'],
        )
    else:
        # Train from scratch
        model = Moco_v2(
            base_encoder=params['backbone'],
            emb_dim=params['emb_dim'],
            use_mlp=params['use_mlp'],
            encoder_momentum=params['encoder_momentum'],
            softmax_temperature=params['softmax_temperature'],
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            num_negatives=params['num_negatives'],
        )
        
    datamodule = datamodules.DecalsDataModule(params)

    # Log various attributes during training
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=params['output_dir'],
        name=file_output_head,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=params['output_dir'],
        filename=file_output_head+'_{epoch:03d}',
        #monitor='train_acc1',
        #mode='max',
        every_n_epochs=params['checkpoint_every_n_epochs'],
        save_top_k=-1,
        save_on_train_epoch_end=True,
        verbose=True,
        save_last=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='epoch',
    )

    if params['strategy'] == 'ddp':
        # DDP set by just Trainer(strategy='ddp') will complain about spending time finding unused parameters
        # Given the model should not have any unused parameters set it explicitly with DDPPlugin
        strategy = DDPPlugin(find_unused_parameters=False)
    else:
        strategy = params['strategy']

    trainer = pl.Trainer.from_argparse_args(
        args,
        strategy=strategy,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
    )

    logger.info("Training Model")
    # Fit model
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=params['ckpt_path']
    )

if __name__=='__main__':

    args = parse_arguments()
    
    main(args)
