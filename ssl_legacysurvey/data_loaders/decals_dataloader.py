import numpy as np
import h5py
import torch
import pytorch_lightning as pl

from torchvision import transforms
from typing import Any, Callable, Optional, List

import os
import glob

from . import decals_augmentations
    
class DecalsDataset(torch.utils.data.Dataset):
    """
    Dataset class for DESI legacy imaging data

    Data can either be in a single hdf5 file, or can be split into chunks over multiple files in the same directory

    Labels can either be loaded from a field in the hdf5 file, or from external arrays of (n_index, n_labels)

    Parameters:
    -----------
    data_path:
        Path to hdf5 files containing galaxy images
    label_path:
        Path to .npy file containing (indices, labels)
    params:
       parameter dictionary
    params['label_name']:
        Field of hdf5 file to use as label. e.g. 'z_phot_median'

    """
    def __init__(
        self,
        data_path,
        label_path,
        transform,
        params,
        max_num_samples=None,
    ):
        self.data_path  = data_path
        self.label_path = label_path
        self.transforms = transform

        self.ssl_training = params.get("ssl_training", False)
        self.label_name = params.get("label_name", None)
        self.predict = params.get("predict", False)
        self.use_ebv = params.get("use_ebv", False)
        self.max_num_samples = max_num_samples if max_num_samples is not None else float('inf')

        # If path is .h5 file, then use that.
        # else assume path is directory containing multiple chunks
        self.multiple_data_files = os.path.splitext(self.data_path)[-1] != ".h5" 
        self.idx_prev = -1

        if self.multiple_data_files:
            # get list of files all sitting in same directory and calculate chunksize
            self.data_files = sorted(glob.glob(self.data_path+"*"))

            with h5py.File(self.data_files[0], 'r') as hf:
                self.chunksize = hf['images'].shape[0]

    def _open_file(self):
        self.hfile = h5py.File(
            self.data_path,
            'r',
        )

    def _open_label_file(self):
        self.labelfile = np.load(self.label_path, mmap_mode='r')

    def __len__(self):

        if not self.ssl_training and self.label_path is not None:
            # Assume labels are in seperate .npy file containing an array of (indices, labels)
            self.n_samples = np.load(self.label_path, mmap_mode='r').shape[0]

        else:
            # During self-supervised training use all data, regardless of whether is has an external label or not
            if not self.multiple_data_files:
                # All data is in a single hdf5 file
                with h5py.File(self.data_path, 'r') as hf:
                    self.n_samples = hf['images'].shape[0]

            else:
                # Calculate total number of data samples across all hdf5 data files
                n_samples = 0
                for ifile, f in enumerate(self.data_files):

                    with h5py.File(f, 'r') as hf:
                        n_samples += hf['images'].shape[0]

                self.n_samples = n_samples

        return int(min(self.n_samples, self.max_num_samples))
        
    def __getitem__(self, idx: int):

        label = idx # Will be overwritten if supervised training specified
        
        if not self.ssl_training and self.label_path is not None:
            # Data is in the form of .npy file containing an array of (indices, labels)
            # Where each index corresponds to the position in the full hdf5 dataset
            # e.g. indices=[0, 62000000] will load the first galaxy in the full dataset, and
            # the 62 million galaxy
            if not hasattr(self, 'labelfile'):
                self._open_label_file()

            label = self.labelfile[idx, 1]
            idx   = int(self.labelfile[idx, 0]) # Now index in decals data, not label file
            
        if not self.multiple_data_files:
            # Data resides in a single hdf5 file given as path
            if not hasattr(self, 'hfile'):
                self._open_file()
            idx_in_chunk = idx
            
        else:
            # Data is split into chunks across multiple hdf5 files, so load idx from correct chunk
            idx_in_file_id = idx // self.chunksize
            idx_prev_in_file_id = self.idx_prev // self.chunksize

            if idx_in_file_id != idx_prev_in_file_id:                
                # Open data file only if it differs from previously opened one
                self.data_path = self.data_files[idx_in_file_id]
                self._open_file()
                
            idx_in_chunk = idx % self.chunksize
            self.idx_prev = idx

        """
        Decals data is in NCHW format so first swap to HWC, 
            then apply augmentatons, then transform back.
        Important: torchvision.to_tensor automatically converts a PIL Image or 
            numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
        """
        im = np.swapaxes(self.hfile['images'][idx_in_chunk], 0, 2) 
         
        if self.use_ebv:
            # Add extinction.
            # decals_augmentations.ebv_to_transmission will use ebv, remove it, and pass along image
            ebv = self.hfile['ebv'][idx_in_chunk]
            im = [im, ebv]      

        # If using Moco2DecalsTransforms im will now be [im_q, im_k]
        # as desired output of dataloader for pl.Moco_v2 is: [image_q, image_k], label 
        # else im will be a single image transform
        im = self.transforms(im) 

        if self.label_name is not None:
            label = self.hfile[self.label_name][idx_in_chunk]
            
        if self.predict:
            return im
        else:
            return im, label 

class DecalsTransforms:
    """
    n_views = 1:
       Returns image with a set of transforms performed
    n_views > 1:
       Returns list of n_views images, each with a set of transforms performed
    """

    def __init__(self, augmentations, params, n_views=1):
        self.n_views = n_views
        
        Augs = decals_augmentations.DecalsAugmentations(augmentations, params)
        
        transform, self.augmentation_names = Augs.add_augmentations()

        transform.append(transforms.ToTensor())
        
        self.transform = transforms.Compose(transform)

    def __call__(self, inp):
        if self.n_views == 1:
            return self.transform(inp)
        else:
            return [self.transform(inp.copy()) for _ in range(self.n_views)]
    
class CropTransform:
    """
    Returns an image with a set of transforms performed
    """

    def __init__(
        self,
        params
    ):
        self.params = params
        self.params['jitter_lim'] = 0
        
        Augs = decals_augmentations.DecalsAugmentations(self.params)
        
        transform, self.augmentation_names = Augs.add_augmentations()

        transform.append(transforms.ToTensor())
        
        self.transform = transforms.Compose(transform)

    def __call__(self, inp):
        q = self.transform(inp)
        return q
