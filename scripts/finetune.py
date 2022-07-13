"""Trains a classification head, or finetune/train from scratch encoder network along with classification head

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
import sys
import glob
from pathlib import Path
import os

import torchvision.models 

from pytorch_lightning import loggers as pl_loggers
from pl_bolts.models.self_supervised import Moco_v2
from pytorch_lightning.plugins import DDPPlugin

from ssl_legacysurvey.data_loaders import datamodules
from ssl_legacysurvey.data_loaders import decals_augmentations
from ssl_legacysurvey.finetune import finetuning

def parse_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='runtime parameters')
    # Data loading
    parser.add_argument("--data_path", type=str, default='/global/cfs/projectdirs/cusp/LBL/decals_galaxy_survey/images/south/',
                        help="Path to hdf5 data file")

    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to hdf5 data file")

    parser.add_argument("--label_path", type=str, default=None,
                        help="Path to labels file. Labels default to these")

    parser.add_argument("--val_label_path", type=str, default=None,
                        help="Path to validation labels file. Labels default to these")

    parser.add_argument("--label_name", type=str, default=None,
                        help="Field name of label in hdf5 file. Will be overidden if using labels from external file")

    parser.add_argument("--max_num_samples", type=int, default=None,
                        help="Maximum number of data samples to use. Defaults to full dataset size")

    parser.add_argument("--num_workers", type=int, default=16,
                        help="number of data loader workers")

    parser.add_argument("--checkpoint_path", type=str, default='../trained_models/test/last.ckpt',
                        help="directory to save trained model and logs")

    # Training
    parser.add_argument("--backbone_weights", type=str, default=None,
                        help="Load from ssl checkpoint, randomly initialize, or use imagenet",
                        choices=['ssl-pretrained', 'random', 'imagenet'])

    parser.add_argument("--finetune", action='store_true',
                        help="Either only train classification head (false), or finetune (true)")
 
    parser.add_argument("--resume_training", action='store_true',
                        help="Resume training from checkpoint, rather than using checkpoint to construct backbone")
       
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU")

    parser.add_argument("--gpus", type=int, default=-1,
                        help="Number of gpus to use")

    parser.add_argument("--num_nodes", type=int, default=1,
                        help="Number of gpu nodes avaialable")

    # ddp does not work in ipython notebook, only ddp_spawn does
    parser.add_argument("--strategy", type=str, default='ddp', #'ddp_spawn',
                        help="training augmentations to use")

    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Run only a few batches")

    parser.add_argument("--max_epochs", type=int, default=10,
                        help="Maximum number of training epochs")
    
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1,
                        help="How often to run validation epoch")
        
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=1,
                        help="Checkpoint model every n epochs")

    parser.add_argument("--save_last", action="store_false",
                        help="Save last epoch")

    parser.add_argument("--num_sanity_val_steps", type=int, default=0,
                        help="Number of validation steps to run right after model initialization")

    # Augmentations
    parser.add_argument("--augmentations", type=str, default='grrrssgbjcgnrg',
                        help="2 character abbreviations of training augmentations to use")

    parser.add_argument("--val_augmentations", type=str, default='ccrg',
                        help="2 character abbreviations of validation augmentations to use")

    parser.add_argument("--jitter_lim", type=int, default=7,
                        help="Number of pixels in x,y to jitter image. (-jitter_lim, jitter_lim)")

    parser.add_argument("--only_dered", action="store_true",
                        help="Deredden if calling gr augmentation")

    parser.add_argument("--only_red", action="store_false",
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
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for model training")

    parser.add_argument("--learning_rate", type=float, default=1e-1,
                        help="Learning rate for classification head")
    
    parser.add_argument("--learning_rate_backbone", type=float, default=1e-3,
                        help="Learning rate for encoder backbone, only used if finetune==True")

    parser.add_argument("--optimizer", type=str, default='Adam',
                        help="Optimizer to use", choices=['Adam', 'SGD'])

    parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR',
                        help="Optimizer to use", choices=['CosineAnnealingLR'])

    parser.add_argument("--T_max", type=int, default=100,
                        help="Cosine decay parameter")

    parser.add_argument("--test_run", action="store_true",
                        help="Subsample training and validation data")

    parser.add_argument("--seed", type=int , default=13579,
                        help="random seed for train test split")

    # Model architecture and settings
    parser.add_argument("--backbone", type=str, default='resnet18',
                        help="Encoder architecture to use", choices=['resnet18', 'resnet50', 'resnet152'])
    
    parser.add_argument("--use_mlp", action="store_true",
                        help="use projection head")

    parser.add_argument("--emb_dim", type=int, default=128,
                        help="Dimensionality where loss is calculated")
    
    parser.add_argument("--num_classes", type=int, default=1,
                        help="Number of output classes")

    parser.add_argument("--prediction_type", type=str, default='regression',
                        help="Specify classification or regression task")

    # Setup outputs and others
    parser.add_argument("--output_dir", type=str, default='../trained_models/test/finetuning/',
                        help="directory to save trained model and logs")

    parser.add_argument("--output_file_tail", type=str, default='',
                        help="String to add to output files")
       
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")

    args = parser.parse_args()

    return args

def main(args):
    """
    Set up model, set training parameters, and configure training callbacks
    """
    params = vars(args) # convert args to dictionary
    params['supervised_training'] = True
    pl.seed_everything(params['seed'], workers=True)

    assert params['label_path'] is not None or params['label_name'] is not None, "if supervised training must specify label_name in hdf5 file or label_path"
            
    Path(params['output_dir']).mkdir(parents=True, exist_ok=True)

    log_name = f"{params['backbone_weights']}{params['output_file_tail']}"
    if not params['finetune']:
        params['learning_rate_backbone'] = 0.0

    log_name += f"_lr{params['learning_rate']}_lrbb{params['learning_rate_backbone']}"
        
    if params['verbose']:
        print("\nTraining with the following parameters:")
        for k, v in params.items():
            print(k, v)

    if not params['resume_training']:
        # If pretrained (or random) CNN backbone is specified then 
        # first load backbone. From local checkpoint in the case of self-supervised pretrained.
        # Then add linear head on output of backbone through model = finetuning.SSLFineTuner(backbone)

        if params['backbone_weights'] in ['imagenet', 'random']:
            # Grab models from torchvision
            backbone = getattr(torchvision.models, params['backbone'])
            backbone = backbone(pretrained=False if params['backbone_weights']=='random' else True)

        if params['backbone_weights'] == 'ssl-pretrained':
            # Load pretrained backbone from disk
            moco_model = Moco_v2.load_from_checkpoint(
                checkpoint_path=params['checkpoint_path'],
            )

            backbone = moco_model.encoder_q
                
        try:
            emb_dim = backbone.fc.in_features # check if model does not use mlp projection head
        except:
            emb_dim = backbone.fc[0].in_features # Model does use mlp projection head

        backbone.fc = torch.nn.Identity() # Remove projection head from model by setting to Identity     


        model = finetuning.SSLFineTuner(
            params=params,
            backbone=backbone,
            in_features=emb_dim,
            num_classes=params['num_classes']

        )
    else:
        model = finetuning.SSLFineTuner.load_from_checkpoint(
            checkpoint_path=params['checkpoint_path'],
        )

    if params['verbose']:
        print(model)

    datamodule = datamodules.DecalsDataModule(params)

    # Log various attributes during training
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(params['output_dir'], 'logs/'),
        name=log_name,
    )

    checkpoint_name = log_name+"_{epoch:02d}_{val_loss:.4f}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=params['output_dir'],
        filename=checkpoint_name,
        monitor="val_loss", #Accuracy",
        mode="min",
        every_n_epochs=params['checkpoint_every_n_epochs'],
        save_top_k=-1,
        save_on_train_epoch_end=True,
        verbose=True,
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = log_name+"_last_{val_Accuracy:.4f}"


    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    '''
    if params['strategy'] == 'ddp':
        # DDP set by just Trainer(strategy='ddp') complain about spending time finding unused parameters
        # Given the model should not have any unused parameters set it explicitly with DDPPlugin
        strategy = DDPPlugin(find_unused_parameters=False)
    else:
        strategy = params['strategy']
    '''
    
    strategy = params['strategy']

    trainer = pl.Trainer.from_argparse_args(
        args,
        strategy=strategy,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
        # auto_lr_find=True,
    )

    # # trainer.lr_find(
    # trainer.tune(
    #     model,
    #     datamodule=datamodule,
    # )
    # print(model.learning_rate)
    

    print("Training Model")
    # Fit model
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=params['checkpoint_path'] if params['resume_training'] else None,
    )

if __name__=='__main__':

    args = parse_arguments()
    
    main(args)
