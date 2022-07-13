"""This module contains various functionalities for dimensionality reduction

Current functionalities:

PCA transform (and inverse transform)
UMAP transform
"""
import math
import glob
import numpy as np
import argparse
import pickle 

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


def pca_transform(data, n_components=8, transform_file_path=None, verbose=False):
    """
    Performs PCA decomposition of input array (N_sample, N_dim). Input array automatically reshaped to (N_sample, -1).
    
    Returns
    -------
    pca_components: 
        (N_sample, ncomponents)
    data_transformed:
        inverse transform of data, keeping only leading PCA components (N_sample, N_dim)
    pca:
        fit pca class
    """
    input_shape = data.shape
    data = data.reshape(input_shape[0], -1)

    pca = PCA(n_components=n_components)
    pca.fit(data)

    pca_components = pca.transform(data)

    data_transformed = pca.inverse_transform(pca_components).reshape(input_shape)

    if transform_file_path:
        print(f'Saving PCA transform to {transform_file_path}')
        with open(transform_file_path, 'wb') as f:
            pickle.dump(pca, f)

        # Test loaded transform returns the same result
        # pca_reload = pickle.load(open(pca_transform_file_path,'rb'))
        # pca_components_new = pca_reload.transform(data)
        # np.testing.assert_array_equal(pca_components, pca_components_new, err_msg='Loaded transform not the same')

    if verbose:
        print('PCA components shape =', pca.components_.shape)
        print('Cumsum explained variance ratio', np.cumsum(pca.explained_variance_ratio_))

    data.reshape(input_shape)

    return pca_components, data_transformed, pca


def umap_transform(data, n_components=2, n_neighbors=50, min_dist=0.5, metric='euclidean', transform_file_path=None, verbose=False):
    """
    Fits UMAP transform to input array of (N_sample, N_dim). Input array automatically reshaped to (N_sample, -1).
    
    Returns
    -------
    umap_components: 
        (N_sample, n_components)
    umap:
        fit UMAP class
    """
    import umap

    input_shape = data.shape
    data = data.reshape(input_shape[0], -1)
     
    umap_trans = umap.UMAP(random_state=0,
                      low_memory=False,
                      n_neighbors=n_neighbors,
                      min_dist=min_dist,
                      n_components=n_components,
                      metric=metric)

    umap_trans.fit(data)
    umap_embedding = umap_trans.transform(data)

    if transform_file_path:
        print(f'Saving UMAP transform to {transform_file_path}')
        with open(transform_file_path, 'wb') as f:
            pickle.dump(umap_trans, f)

        # Test loaded transform returns the same result
        # umap_reload = pickle.load(open(umap_transform_file_path,'rb'))
        # umap_embedding_new = umap_reload.transform(data)
        # np.testing.assert_array_equal(umap_embedding, umap_embedding_new, err_msg='Loaded transform not the same')

    if verbose:
        print(f"UMAP embedding is shape {umap_embedding.shape}")

    data.reshape(input_shape)

    return umap_embedding, umap_trans


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='runtime parameters')


    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of data samples")
 
    parser.add_argument("--sample_dimensionality", type=int, default=100,
                        help="Dimensionality of each data sample")
       
    parser.add_argument("--n_pca_components", type=int, default=8,
                        help="Number of leading PCA components to calculate/keep")

    parser.add_argument("--n_umap_components", type=int, default=2,
                        help="Output dimensionality of UMAP")
       
    args = parser.parse_args()

    return args


def main(params: dict):

    np.random.seed(13579)
    data = np.random.normal(0, 1, size=(params['n_samples'], 10, params['sample_dimensionality']))

    print('\nTesting PCA\n')
    pca_components, data_transformed, pca = pca_transform(
        data,
        n_components=params['n_pca_components'],
        transform_file_path='test_pca_transform.pkl'
        ) 

    print(f"Input data of dim: {data.shape}")
    print(f"Kept {params['n_pca_components']} pca components ({pca_components.shape}), for resulting reconstruction shape of {data_transformed.shape}")

    print('\nTesting UMAP\n')
    umap_embedding, umap_trans = umap_transform(
        data,
        n_components=params['n_umap_components'],
        transform_file_path='test_UMAP_transform.pkl'
        ) 

    print(f"Input data of dim: {data.shape}")
    print(f"Resulting in UMAP embedding of shape {umap_embedding.shape}")


if __name__ == '__main__':

    args = parse_arguments()
    params = vars(args)

    main(params)