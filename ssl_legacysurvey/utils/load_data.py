"""
load_data.py contains DecalsDataLoader class with three main functionalities.

Given a list or array of indices (from 0 to N_galaxy - 1):

Load galaxy catalogue information

Load images

Load representations

"""

import h5py
import numpy as np
import os
import math
import glob
import logging
from scipy import interpolate

def flux_to_mag(flux):
    """convert from flux in nanomaggies to magnitudes"""

    return 22.5 - 2.5*np.log10(flux)

def select_indices_vs_mag(DDL, n_sample=5000, magz_min=6., magz_max=20.):
    """
    Select galaxies to load in uniformly as a function of magnitude
    
    Returns
    -------
    indices of galaxies in full dataset
    """
    # Load all galaxies
    inds_all = np.arange(DDL.ngals_tot)
    print("\nloading flux of all galaxies in survey. Be patient.\n")

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

    zmag_select = np.linspace(magz_min, magz_max, n_sample)# zmag_select = np.logspace(np.log10(zmin), np.log10(zmax), 10000)

    inds_use = np.unique(f(zmag_select).astype(np.int))

    return inds_use

class DecalsDataLoader:
    """
    Class to retrieve images, sweep catalogue fields, or representations.

    Data is loaded based on index array or list which has values ranging from 0 to ngals_tot-1.
    """

    def __init__(self, survey='south',
                 data_dir='/global/cfs/projectdirs/cusp/LBL/decals_galaxy_survey/',
                 image_dir=None,
                 rep_dir=None,
                 field_dir=None,
                 rep_file_head=None,
                 npix_in=152,
                 verbose=False):

        self.band_labels = {0: 'g',
                       1: 'r',
                       2: 'z'}

        self.survey = survey
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.rep_dir = rep_dir
        if image_dir is None:
            self.image_dir = f'{data_dir}/images/{survey}/'

        # Determine info about galaxy/image dataset
        # Likely spread across h5py files, each of a uniform chunksize (except for the last file) 
        self.data_files = sorted(glob.glob(self.image_dir+'*'))
        self.lead_file = self.data_files[0]
        self.ngals_tot = 0
        for i, f in enumerate(self.data_files):
            with h5py.File(f, 'r') as hf:
                if i == 0: 
                    self.fields_available = sorted(list(hf.keys()))
                    self.chunksize = hf['images'].shape[0]
                    self.npix_in = hf['images'].shape[-1]
                else:
                    self.fields_available =  sorted(list(set(self.fields_available) & set(list(hf.keys()))))

                for field in self.fields_available:
                    if hf[field].shape[0] != hf['images'].shape[0]: # deal with any missing data
                        self.fields_available.remove(field)

                self.ngals_tot += hf['images'].shape[0]

        # Determine info about representations
        # Spread across .npy files, each of a uniform chunksize (except for the last file) 
        if self.rep_dir:
            self.rep_files = sorted(glob.glob(self.rep_dir+'*'))
            self.nrep_tot = 0
            for i, f in enumerate(self.rep_files):
                data = np.load(f, mmap_mode='r')
                if i == 0: 
                    self.chunksize_rep = data.shape[0]
                    self.ndim_representation = data.shape[1]

                self.nrep_tot += data.shape[0]
                   
            self.fields_available += 'representations'
            self.nchunks_rep = int(math.ceil(self.nrep_tot/self.chunksize_rep))

        if verbose:
            print(f"\nTotal number of samples in dataset: {self.ngals_tot}")
            print(f"\nFields available to load: {self.fields_available}")       
        self.nchunks = int(math.ceil(self.ngals_tot/self.chunksize))

        self.verbose = verbose

    def get_data(self, inds, fields='images', want_mag=True, npix_out=152):
        """
        Return data corresponding to index array/list

        Attributes
        ----------
        inds : array or list
            index values to load in
        fields : string or list 
            data field(s) desired. e.g. 'images', 'representations', ['ra', 'dec', ...] 
        """
        if isinstance(inds, int) and inds==-1:
            inds = np.arange(self.ngals_tot)

        inds = np.array(inds)
        ngals_i = inds.shape[0]

        if type(fields)==str:
            fields = fields.split()

        npix_start = self.npix_in//2 - npix_out//2
        npix_end = self.npix_in//2 + npix_out//2

        dataset = {}
        dataset['inds'] = inds

        for field in fields:
            field_out = field
            if field == 'flux' and want_mag:
                field_out = 'mag'

            # Set up numpy arrays to load data into
            if field == 'representations':
                data_out = np.empty((ngals_i, self.ndim_representation), dtype=np.float32)

            else:
                with h5py.File(self.lead_file, 'r') as hf:
                    field_shape = hf[field][0].shape
                    field_dtype = hf[field].dtype

                if field == 'images':
                    field_shape = (field_shape[0], npix_out, npix_out)

                data_out = np.empty(([ngals_i]+[dim for dim in field_shape]), dtype=field_dtype)

            num_data_loaded = 0
            nchunk = self.nchunks
            if field == 'representations':
                nchunk = self.nchunks_rep
            for ichunk in range(nchunk):
                # Load in data from each chunk on disk
                if num_data_loaded == ngals_i:
                    break

                if self.verbose: logging.info(f"Loading {fields} from chunk: {ichunk}")

                igal_start = ichunk*self.chunksize
                igal_end = min(igal_start+self.chunksize, self.ngals_tot)

                if field == 'representations':
                    # representations are generally stored separate from original data
                    data_file = np.load(self.rep_files[ichunk], mmap_mode='r')

                else:
                    # Images and galaxy info are stored in the same .hdf5 files
                    data_file = self.data_files[ichunk]

                # Data must be liaded from hdf5 files in order of increasing index
                dm = (inds >= igal_start) & (inds < igal_end)
                ind_i = inds[dm] % self.chunksize
                lin_inds = np.arange(dm.shape[0])[dm]
                
                ind_sort = np.argsort(ind_i)
                ind_i = ind_i[ind_sort]
                lin_inds = lin_inds[ind_sort]

                num_data_loaded += ind_i.shape[0]
                if ind_i.shape[0] > 0:
                    if field == 'representations':
                        data_out[lin_inds] = data_file[ind_i]

                    else:
                        with h5py.File(data_file, 'r') as f:
                            if field == 'images':
                                data_out[lin_inds] = f['images'][ind_i, :, npix_start:npix_end, npix_start:npix_end]
                            else: 
                                data_out[lin_inds] = f[field][sorted(ind_i)]

            if data_out.ndim != 2 or field=='representation':
                dataset[field] = data_out

            else :   
                # seperate values for each band (g, r, z), so split
                for iband in range(data_out.shape[1]):
                    if field=='flux' and want_mag:
                        data_out[:, iband] = flux_to_mag(data_out[:, iband])

                    dataset[f"{field_out}_{self.band_labels[iband]}"] = data_out[:, iband]
                
        return dataset


if __name__=="__main__":

    ngals = 8
    image_dir = '/global/cfs/projectdirs/cusp/LBL/decals_galaxy_survey/images/south/'

    DDL = DecalsDataLoader(verbose=True, image_dir=image_dir)

    # Choose <ngals> by random indices from [0, dataset_size)
    inds_use = np.random.choice(DDL.ngals_tot, ngals, replace=False)

    fields = ['images', 'ra', 'flux', 'source_type'] 

    galaxies = DDL.get_data(inds_use, fields=fields)
    for k in galaxies:
        print(f"{k} shape:", galaxies[k].shape)
