import h5py
import numpy as np
import matplotlib.pyplot as plt

import os
import math 
import glob
import pandas as pd 
import argparse
from pathlib import Path

#from ssl_legacysurvey.src                                                                                               
from ssl_legacysurvey.utils import load_data
from ssl_legacysurvey.utils import plotting_tools as plt_tools
from ssl_legacysurvey.data_analysis import dimensionality_reduction
from ssl_legacysurvey.data_analysis import select_galaxies

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='runtime parameters')

    parser.add_argument("--train_pca", action="store_true",
                        help="Train and use pca_transform")

    parser.add_argument("--train_umap", action="store_true",
                        help="Train and use umap_transform")

    parser.add_argument("--pca_pre_umap", action="store_true",
                        help="Decompose data into PCA components, then perform UMAP")

    parser.add_argument("--nsamples", type=int, default=10000,
                        help="Number of data samples. Might be automatically downsampled due to selecting as a function of magnitude")
 
    parser.add_argument("--n_pca_components", type=int, default=8,
                        help="Number of leading PCA components to calculate/keep")

    parser.add_argument("--n_umap_components", type=int, default=2,
                        help="Output dimensionality of UMAP")

    parser.add_argument("--distance_metric", type=str, default='cosine',
                        help="Distance metric to use for UMAP")

    parser.add_argument("--ndim_representation", type=int, default=512,
                        help="Dimensionality of representations")

    parser.add_argument("--output_dir", type=str, default='../data/analysis/',
                        help="file to save the PCA transform in")

    parser.add_argument("--pca_file_head", type=str, default='pca',
                        help="file to save the PCA transform in")

    parser.add_argument("--umap_file_head", type=str, default='umap',
                        help="file to save the UMAP transform in")

    parser.add_argument("--representation_directory", type=str, default='../trained_models/resnet18_old/representations/',
                        help="Directory containing image representations")

    parser.add_argument("--representation_file_head", type=str, default='model_outputs',
                        help="File head of representations (i.e. <filehead>_00000000_01000000")
                           
    args = parser.parse_args()

    return args


def main(params):

    survey = 'south'
    to_rgb = True

    Path(params['output_dir']).mkdir(parents=True, exist_ok=True)

    # Class to load in images/representations
    DDL = load_data.DecalsDataLoader(
        survey=survey,
        to_rgb=to_rgb,
        rep_dir=params['representation_directory'],
        rep_file_head=params['representation_file_head'],
        ndim_representation=params['ndim_representation'])

    # Indices of galaxies in the full Decals dataset
    inds_use = select_galaxies.select_indices_vs_mag(DDL, params['nsamples'])
    np.random.shuffle(inds_use)
    print(f"Kept {len(inds_use)} galaxies after subselecting in magnitude")

    rep = DDL.get_representations(inds_use)

    params['pca_transform_file_path'] = os.path.join(params['output_dir'], f"{params['pca_file_head']}_{len(inds_use)}_transform.pkl")
    params['umap_transform_file_path'] = os.path.join(params['output_dir'], f"{params['umap_file_head']}_{len(inds_use)}_transform.pkl")

    params['pca_embedding_file_path'] = os.path.join(params['output_dir'], f"{params['pca_file_head']}_{len(inds_use)}_embedding.npz")
    params['umap_embedding_file_path'] = os.path.join(params['output_dir'], f"{params['umap_file_head']}_{len(inds_use)}_embedding.npz")

    if params['train_pca'] or params['pca_pre_umap']:
        # Train and save PCA decomposition
        pca_components, rep_pca, pca = dimensionality_reduction.pca_transform(
            rep,
            n_components=params['n_pca_components'],
            transform_file_path=params['pca_transform_file_path'],
            )

        np.savez(
            params['pca_embedding_file_path'],
            data=rep,
            data_recon=rep_pca,
            embedding=pca_components,
            decals_inds=inds_use,
            )

    if params['train_umap']:
        # Train and save UMAP transform
        if params['pca_pre_umap']:
            rep = pca_components
            print('PCA shape', rep.shape)

        umap_embedding, umap_trans = dimensionality_reduction.umap_transform(
            rep,
            n_components=params['n_umap_components'],
            metric=params['distance_metric'],
            transform_file_path=params['umap_transform_file_path'],
            )

        np.savez(
            params['umap_embedding_file_path'],
            data=rep,
            embedding=umap_embedding,
            decals_inds=inds_use,
            )

if __name__ == '__main__':

    args = parse_arguments()
    params = vars(args)

    main(params)