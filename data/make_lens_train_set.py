"""Creates training sets for Galaxy Zoo morphological classification questions"""

import numpy as	np
import pandas as pd
import h5py

from ssl_legacysurvey.utils import load_data
from ssl_legacysurvey.data_analysis import select_galaxies

import argparse

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='runtime parameters')
    

    # Data loading
    parser.add_argument("--data_path", type=str, default='/global/homes/g/gstein/src/ssl-legacysurvey/data/south_lenses_all.tsv',
                        help="Path to lens data file")

    parser.add_argument("--max_num_labels", type=int, default=-1,
                        help="Use only use this number of labels for train/validation. Test set remains a fraction of the original total")
 
    parser.add_argument("--train_frac", type=float, default=0.9,
                        help="Fraction of non-test galaxies to use in the training set. Validation set receives the remaining")
 
    # parser.add_argument("--val_frac", type=float, default=0.1,
    #                     help="Fraction of galaxies to use in the validation set")
  
    parser.add_argument("--test_frac", type=float, default=0.1,
                        help="Fraction of galaxies to use in the test set")
         
    parser.add_argument("--num_negative_samples", type=int, default=1000,
                        help="Number of negative samples to use in training set")

    parser.add_argument("--seed", type=int, default=13579,
                        help="Random seed")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")

    args = parser.parse_args()

    return args

def save_to_disk(inds, labels, file_head):
	"""Loads in images and galaxy data from array of indices, and saves all to h5py file. 

	Also saves indices and labels in own .npy file
	"""

	print(f"saving {file_head}")
	np.save(f"{file_head}.npy", np.c_[inds, labels])

	survey = 'south'

	DDL = load_data.DecalsDataLoader(survey=survey)

	gals = DDL.get_data(inds, fields=DDL.fields_available, npix_out=152)

	with h5py.File(f"{file_head}.h5", "w") as f:
	    for k, v in gals.items():
	        if v.dtype=='float32' or v.dtype=='int32' or v.dtype=='float64' or v.dtype=='int64':
	            f.create_dataset(k, data=v)
	        
	    f.create_dataset('labels', data=labels)

def main(args):

	params = vars(args) # convert args to dictionary

	np.random.seed(params['seed'])

	# First load lenses
	df = pd.read_csv(params['data_path'], sep='\t')
	inds_lens = df['inds'].to_numpy()

	print(f"Number of lens samples : {inds_lens.shape[0]}")

	if params['verbose']:
		for col in df.columns:
		    print(col)

	# Select non-lenses
	survey = 'south'

	# Class to load in images/representations
	DDL = load_data.DecalsDataLoader(survey=survey)

	# Indices of galaxies in the full Decals dataset
	inds_nonlens = select_galaxies.select_indices_vs_mag(DDL, params['num_negative_samples'])
	inds_nonlens = np.array(list(set(inds_nonlens) - set(inds_lens))) # remove any that are actually positive samples
	print(f"Number of non-lens samples after magnitude selection: {inds_nonlens.shape[0]}")
	# Extract relevant columns from large Pandas dataframe

	inds_decals = np.concatenate((inds_lens, inds_nonlens))
	labels = np.zeros(inds_decals.shape[0])
	labels[:inds_lens.shape[0]] = 1
	
	print(f"Fraction of lenses = {labels.mean()}")
	# Train/Val/Test split
	print(f"Total number of samples: {len(labels)}")
	inds_lin = np.arange(labels.shape[0])
	np.random.shuffle(inds_lin)

	ntrain_val_max = int(inds_lin.shape[0] * (1 - params['test_frac']))
	if params['max_num_labels'] > 0:
		ntrain_val_max = min(ntrain_val_max, params['max_num_labels'])

	ntrain = int(ntrain_val_max * params['train_frac'])
	nval = ntrain_val_max - ntrain 
	ntest = int(inds_lin.shape[0]*params['test_frac'])

	inds_train = inds_lin[:ntrain]
	inds_val   = inds_lin[ntrain:ntrain+nval]
	inds_test = inds_lin[-ntest:]
	print(f"Ntrain = {len(inds_train)}, Nval = {len(inds_val)}, Ntest = {len(inds_test)}")

	labels_train, labels_val, labels_test = labels[inds_train], labels[inds_val], labels[inds_test]
	inds_train, inds_val, inds_test = inds_decals[inds_train], inds_decals[inds_val], inds_decals[inds_test]

	save_to_disk(inds_train, labels_train, f"../../data/lens_labels_train_{len(inds_train)}")
	save_to_disk(inds_val, labels_val, f"../../data/lens_labels_val_{len(inds_val)}")
	save_to_disk(inds_test, labels_test, f"../../data/lens_labels_test_{len(inds_test)}")

if __name__ == '__main__':

	args = parse_arguments()

	main(args)
