import numpy as np
import h5py
import torch
import pytorch_lightning as pl

from torchvision import transforms
from typing import Any, Callable, Optional, List

import os
import glob

from . import decals_augmentations
from .decals_dataloader import *

class DecalsDataModule(pl.LightningDataModule):
    """
    Loads data from either a single large hdf5 file,
    or from a set of hdf5 files located in the same directory
    """
    
    def __init__(
        self,
        params: dict,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
    ) -> None:
        
        super().__init__()
        
        self.params = params
        self.ssl_training = self.params.get("ssl_training", False)

        self.data_path = self.params.get("data_path", "./")
        self.label_path = self.params.get("label_path", None)
        self.augmentations = self.params.get("augmentations", None)

        self.val_data_path = self.params.get("val_data_path", self.data_path) # Set val to same as train if unspecified
        self.val_label_path = self.params.get("val_label_path", None)
        self.val_augmentations = self.params.get("val_augmentations", 'cc')

        self.num_workers = self.params.get("num_workers", 1)
        self.batch_size = self.params.get("batch_size", 4)
        
        self.shuffle = self.params.get("shuffle", True)
        self.pin_memory = self.params.get("pin_memory", True)
 
        if self.ssl_training:
            self.n_views = 2
            self.drop_last = True # drop last due to queue_size % batch_size == 0. assert in Moco_v2
        else:      
            self.n_views = 1
            self.drop_last = False

    def _default_transforms(self, training=True) -> Callable:
        """
        n_views = 1: Returns [im], label
        n_views = 2: Returns [im_k, im_q], label
        """
        augs = self.augmentations if training else self.val_augmentations

        transform = DecalsTransforms(augs, self.params, n_views=self.n_views)

        return transform    
    
    def prepare_data(self) -> None:

        # If path is .h5 file, then use that.
        # Else assume path is directory containing multiple chunks
        multiple_data_files = os.path.splitext(self.data_path)[-1] != ".h5"

        if not multiple_data_files and not os.path.isfile(self.data_path):
            raise FileNotFoundError(
                """
                Your training datafile cannot be found
                """
            )
        if multiple_data_files:
            self.data_files = sorted(glob.glob(self.data_path+"*"))
            if len(self.data_files) < 1:
                raise FileNotFoundError(
                    """
                    No files found in specified data directory
                    """
                )   
                   
    def setup(self, stage: Optional[str] = None):
        
        # Assign train/val datasets for use in dataloaders     
        if stage == "fit" or stage is None:
    
            train_transforms = self._default_transforms(training=True) if self.train_transforms is None else self.train_transforms
            self.train_dataset = DecalsDataset(
                self.data_path,
                self.label_path,
                train_transforms,
                self.params,
                max_num_samples=self.params['max_num_samples'],  # subsample train dataset
            )
            
            # Val set not used for self-supervised pretraining
            val_transforms = self._default_transforms(training=False) if self.val_transforms is None else self.val_transforms
            self.val_dataset = DecalsDataset(
                self.val_data_path,
                self.val_label_path,
                val_transforms,
                self.params,
                max_num_samples=None, # use fill val dataset
            )

        if stage == "predict" or stage is None:

            # predict_transforms = CropTransform(self.params)
            predict_transforms = self._default_transforms(training=False) if self.val_transforms is None else self.val_transforms
            # Predict over all training data
            self.predict_dataset = DecalsDataset(
                self.data_path,
                self.label_path,
                predict_transforms,
                self.params,
                max_num_samples=None,  # use fill test dataset
            )
                    
    def train_dataloader(self):
 
        loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
            
        return loader
    
    def val_dataloader(self):

        loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )       
        
        return loader

    def predict_dataloader(self):
        
        loader = torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )       
        
        return loader
