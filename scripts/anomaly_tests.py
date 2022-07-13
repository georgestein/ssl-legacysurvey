import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import glob

#from ssl_legacysurvey.src
import src.data_loader as data_loader

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

num_pca_keep = 10

def pca_transform(data, ncomponents=2):
    pca = PCA(n_components=ncomponents)
    pca.fit(rep)

    pca_components = pca.transform(rep)
    rep_new = pca.inverse_transform(pca_components)

    return pca_components, rep_new, pca

if __name__ == '__main__':

    DDL = data_loader.DecalsDataLoader()
    
    inds_use = np.arange(1000)

    rep_file = 'rep_test.npy'
    if not os.path.exists(rep_file):
        
        rep = DDL.get_representations(inds_use)    
        np.save(rep_file, rep)

    else:
        rep = np.load(rep_file)
        print(rep)

    # PCA decomposition
    pca_components, rep_new, pca = pca_transform(rep, ncomponents=num_pca_keep)

    print(pca.components_.shape)
    print(pca.explained_variance_ratio_)
    print(np.cumsum(pca.explained_variance_ratio_))

    # Isolation forest
    clf = IsolationForest(max_samples=100,
                          random_state=0)
    clf.fit(pca_components)
    
    print(clf.predict(pca_components).shape)
