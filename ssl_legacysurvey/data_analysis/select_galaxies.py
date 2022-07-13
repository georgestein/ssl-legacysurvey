"""Select galaxies weighted towards lower magnitudes"""

import numpy as np
from scipy import interpolate

def select_indices_vs_mag(DDL, n_sample=5000, zmin=6., zmax=20.):
    """
    Select galaxies to load in uniformly as a f magnitude
    
    Returns
    -------
    anomaly_score: 
        (N_sample)
    clf:
        fit isolation forest class
    """
    # Load all galaxies
    inds_all = np.arange(DDL.ngals_tot)
    gals = DDL.get_data(inds_all, fields=['flux'])

    # Sort by magnitude
    inds_sort = np.argsort(gals['mag_z'])
    mag_sort = gals['mag_z'][inds_sort]
    inds_all_sort = inds_all[inds_sort]

    # mag_sort[mag_sort > zmax] = zmax
    # mag_sort[mag_sort < zmin] = zmin

    # Select new galaxies as a function of magnitude
    inds_lin = np.arange(mag_sort.shape[0])
    f = interpolate.interp1d(mag_sort, inds_lin, 
                             bounds_error=False, 
                             fill_value=inds_lin[inds_lin.shape[0]//2])

    zmag_select = np.linspace(zmin, zmax, n_sample)# zmag_select = np.logspace(np.log10(zmin), np.log10(zmax), 10000)

    inds_use = np.unique(f(zmag_select).astype(np.int))

    return inds_use
