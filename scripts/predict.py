"""Use trained model to predict over dataset.

Can be used to output classification predictions, or to extract representations:
--extract_representations==True
    returns outputs of CNN backbone, before MLP prediction head
--extract_representations==False
    returns outputs of backbone+linear classification head

Network outputs are extracted through a PyTorch-Lightning callback (extract_model_outputs.OutputWriter).

As the dataset size is very large, they are first saved to disk in individual batch files.
Once all batches are output to disk they are then compiled into final file(s) of size <chunksize>.

Batched outputs can be found in <output_dir>/chunks, while compiled outputs are found at <output_dir>/compiled

If <label_path> or <val_label_path> are specified then model will only be called only those subsets of the total dataset
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
import math

from pytorch_lightning import loggers as pl_loggers
from pl_bolts.models.self_supervised import Moco_v2
from pytorch_lightning.plugins import DDPPlugin

from ssl_legacysurvey.data_loaders import datamodules
from ssl_legacysurvey.data_loaders import decals_augmentations
from ssl_legacysurvey.finetune import finetuning
from ssl_legacysurvey.finetune import extract_model_outputs

def parse_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='runtime parameters')
    # Data loading
    parser.add_argument("--data_path", type=str, default='/global/cfs/projectdirs/cusp/LBL/decals_galaxy_survey/images/south/',
                        help="Path to hdf5 data file")

    parser.add_argument("--label_path", type=str, default=None,
                        help="Path to labels file. Labels default to these")

    parser.add_argument("--val_label_path", type=str, default=None,
                        help="Path to validation labels file. Labels default to these")

    parser.add_argument("--label_name", type=str, default=None,
                        help="Field name of label in hdf5 file. Will be overidden if using labels from external file")

    parser.add_argument("--num_workers", type=int, default=16,
                        help="number of data loader workers")

    parser.add_argument("--checkpoint_path", type=str, default='../trained_models/test/last.ckpt',
                        help="directory to save trained model and logs")

    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU")

    parser.add_argument("--gpus", type=int, default=-1,
                        help="Number of gpus to use")

    parser.add_argument("--num_nodes", type=int, default=1,
                        help="Number of gpu nodes available")

    # ddp does not work in ipython notebook, only ddp_spawn does
    parser.add_argument("--strategy", type=str, default='ddp', #'ddp_spawn',
                        help="Training strategy to use")

    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Run only a few batches")

    parser.add_argument("--max_epochs", type=int, default=1,
                        help="Max epochs to predict (1)")

    # Augmentations
    parser.add_argument("--augmentations", type=str, default='ccrg',
                        help="2 character abbreviations of training augmentations to use")

    parser.add_argument("--val_augmentations", type=str, default='ccrg',
                        help="2 character abbreviations of validation augmentations to use")

    parser.add_argument("--jitter_lim", type=int, default=0,
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

    parser.add_argument("--test_run", action="store_true",
                        help="Subsample training and validation data")

    # Model architecture and settings
    parser.add_argument("--backbone", type=str, default='resnet18',
                        help="Encoder architecture to use", choices=['resnet18', 'resnet50', 'resnet152'])

    parser.add_argument("--use_mlp", action="store_true",
                        help="use projection head")

    parser.add_argument("--use_mlp_representation", action="store_true",
                        help="Output post-projection head representations")

    parser.add_argument("--emb_dim", type=int, default=128,
                        help="Dimensionality where loss is calculated")

    parser.add_argument("--num_classes", type=int, default=1,
                        help="Number of output classes")

    parser.add_argument("--prediction_type", type=str, default='classification',
                        help="Specify classification or regression task")

    # Setup outputs and others
    parser.add_argument("--output_dir", type=str, default='../trained_models/test/finetuning/',
                        help="directory to save trained model and logs")

    parser.add_argument("--chunksize", type=int, default=1000000,
                        help="Chunksize of desired output files")

    parser.add_argument("--max_num_samples", type=int, default=float('inf'),
                        help="Max number of samples to use")

    parser.add_argument("--file_head", type=str, default='model_outputs',
                        help="head of filepath")
 
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite any predictions on disk at same location")
  
    parser.add_argument("--only_compile", action="store_true",
                        help="Don't extract representations, only compile")
            
    parser.add_argument("--extract_representations", action="store_true",
                        help="Output post-projection head representations, rather than model predictions")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")

    args = parser.parse_args()

    return args

def compile_chunk(params, all_batch_files, ichunk):
    """
    Load in data from each batch output by each gpu,
    Assemble in proper order according to index,
    and save as larger ordered chunks.

    If saved using a single GPU output indices will already be in order,
    but if saved using e.g. DDP on 2 GPUs will alternate indices:

    batch_0000000_000.npz: batch from gpu 0, holds data from indices [0,2,4,6,...] 
    batch_0000000_001.npz: batch from gpu 1, holds data from indices [1,3,5,7,...] 
    """
    
    igal_start = ichunk*params['chunksize']
    igal_end   = min(params['ngals_tot'], (ichunk+1)*params['chunksize'])

    ibatch_start = igal_start // params['batch_size']
    ibatch_end = int(math.ceil(igal_end / params['batch_size']))
    
    ifile_start = ibatch_start * params['num_gpus']
    ifile_end = ibatch_end * params['num_gpus']

    this_chunksize = igal_end - igal_start

    # ensure output directory exists
    compiled_dir = os.path.join(params['output_dir'], 'compiled/')
    Path(compiled_dir).mkdir(parents=True, exist_ok=True)

    file_out_path = os.path.join(
        compiled_dir,
        f"{params['file_head']}_{igal_start:09d}_{igal_end:09d}.npy",
        )

    if params['verbose']:
        print(igal_start, igal_end, ibatch_start, ibatch_end, ifile_start, ifile_end)

    data_all = np.empty(
        (this_chunksize, params['data_dim']),
        dtype=np.float32,
    )
    
    for i, f in enumerate(all_batch_files[ifile_start:ifile_end]):
        if params['verbose'] and (i % 10000)==0:
            print(f"{i}: Loading file {f}")
        batchi, gpui = os.path.splitext(f)[0].split('_')[-2:]
        batchi, gpui = int(batchi), int(gpui)

        # print(f, batchi, gpui)

        data = np.load(f)
        
        batch_size = data['data'].shape[0]
        if i == 0:
            global_batch_size = int(batch_size*params['num_gpus'])


        # if params['verbose']:
        #     print(data['batch_indices'])

        if data['batch_indices'].size > 0:
            # If batch indices were saved alongside predictions then use those
            batch_indices = data['batch_indices']
        else:
            # Else we need to reconstruct what the indices each prediction belongs to.
            # To do so we use num_gpus and strategy, because PyTorch-Lightning currently has a bug when
            # Trainer.predict(return_predictions=False) (see my issue at https://github.com/Lightning-AI/lightning/issues/13580)
            # So for now I implement an after-the-fact fix here.
            if params['strategy']=='ddp':
                # if using 2 devices and batch_size=4
                # device 0: [0, 2]
                # device 1: [1, 3]
                batch_indices = np.arange(0, batch_size*params['num_gpus'], params['num_gpus']) + gpui
                batch_indices += batchi * global_batch_size

            else:
                sys.exit(f"Indices were not found in prediction files" 
                    +" and insure of matching between indices and predictions when using strategy={params['strategy']}")

        inds_keep = (batch_indices >= igal_start) & (batch_indices < igal_end)

        data_all[batch_indices[inds_keep] % params['chunksize']] = data['data'][inds_keep]

    np.save(file_out_path, data_all)

def compile_chunks(params):

    all_batch_files = sorted(glob.glob(os.path.join(params['output_dir'], f"chunks/{params['file_head']}*")))

    # Determine batch size (from length of data array) and number of GPUS used.
    # data arrays are (N, ndim)
    f_ = np.load(all_batch_files[0])
    print(f"\n{len(all_batch_files)} batches total. Files in each batch .npz are: ", f_.files)
    params['batch_size_per_gpu'] = f_['data'].shape[0]
    params['data_dim'] = f_['data'].shape[1]
    params['num_gpus'] = f_['num_gpus']

    params['batch_size'] = int(params['batch_size_per_gpu'] * params['num_gpus'])

    # find total number of predictions in dataset
    # batches will all be the same size until (possibly) the final one
    ngals_tot = params['batch_size_per_gpu'] * (len(all_batch_files) - params['num_gpus']) # add full batches

    for i, f in enumerate(all_batch_files[-params['num_gpus']:]):
        # add last batch

        # if (i % 1000)==0:
        #     print(f"Getting shape from file chunk {i} of {len(all_batch_files)}")
        ngals_tot += np.load(f, mmap_mode='r')['data'].shape[0]

    print(ngals_tot)
    params['ngals_tot'] = ngals_tot
    params['chunksize'] = min(params['chunksize'], ngals_tot)

    print(f"\nBatched data of dimensionality {params['data_dim']} \
        was extracted with batch size per gpu={params['batch_size_per_gpu']} \
        using {params['num_gpus']} GPUs")

    # Compile all chunks
    nchunks = int(math.ceil(params['ngals_tot']/params['chunksize']))
    for ichunk in range(61, nchunks):
        print('Compiling data in chunk: ', ichunk)

        compile_chunk(params, all_batch_files, ichunk)


def main(args):
    """
    Set up model and predict over dataset
    """
    params = vars(args) # convert args to dictionary
    params['supervised_training'] = True
    Path(params['output_dir']).mkdir(parents=True, exist_ok=True)
    params['predict'] = not params['extract_representations'] # if not extracting representations then must want model predictions

    if params['verbose']:
        print("\nTraining with the following parameters:")
        for k, v in params.items():
            print(k, v)


    if params['extract_representations']: 
        # Do not need any classification head, just SSL trained model
        moco_model = Moco_v2.load_from_checkpoint(
            checkpoint_path=os.path.join(params['checkpoint_path'])
            )

        # extract encoder_q from Moco_v2 model
        backbone = moco_model.encoder_q

        if not params['use_mlp_representation']:

            try:
                # check if model does not use mlp projection head
                emb_dim = backbone.fc.in_features
            except:
                # Model does use mlp projection head
                emb_dim = backbone.fc[0].in_features

            # Remove projection head by setting to Identity layer
            backbone.fc = torch.nn.Identity()

        else:
            emb_dim = params['emb_dim']

        print(f"Representation dimensionality to be extracted: {emb_dim}")

        model = extract_model_outputs.OutputExtractor(backbone)

    else:
        model = finetuning.SSLFineTuner.load_from_checkpoint(
            checkpoint_path=params['checkpoint_path'],
            )

    if params['verbose']:
        print(model)

    datamodule = datamodules.DecalsDataModule(params)
    datamodule.setup(stage="predict")

    # Predictions are written during trainer.predict by this callback
    prediction_dir = os.path.join(params['output_dir'], 'chunks/')
    Path(prediction_dir).mkdir(parents=True, exist_ok=True)

    # Clear previous files so chunks don't get incorrectly compiled
    previous_files = glob.glob(prediction_dir + "*")
    # for f in previous_files:
    #     os.remove(f)

    PredictionWriterCallback = extract_model_outputs.OutputWriter(
        prediction_dir,
        file_head=params['file_head'],
        overwrite=params['overwrite'],
        write_interval="batch",
    )
    
    strategy = params['strategy']

    trainer = pl.Trainer.from_argparse_args(
        args,
        strategy=strategy,
        callbacks=[PredictionWriterCallback],
    )

    print("Predicting over dataset")
    #Predict over dataset and save each batch to disk                 
    # trainer.predict(
    #     model,
    #     datamodule=datamodule,
    #     return_predictions=False, # On such a large dataset storing full predictions will crash, so write in batches
    # )

    print("DEBUG", torch.cuda.current_device())


    # Compile batches into final file
    if torch.cuda.current_device() == 0:
        compile_chunks(params)

if __name__=='__main__':

    args = parse_arguments()
    
    if args.only_compile:
        params = vars(args)
        compile_chunks(params)

    else:
        main(args)
