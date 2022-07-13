"""Creates training sets for Galaxy Zoo morphological classification questions"""

import numpy as	np
import pandas as pd
import h5py

from ssl_legacysurvey.utils import load_data

import argparse

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='runtime parameters')
    

    # Data loading
    parser.add_argument("--data_path", type=str, default='/global/cfs/projectdirs/cusp/LBL/decals_galaxy_survey/external_catalogues/galaxy_zoo_decals/gz_decals_volunteers.csv',
                        help="Path to hdf5 data file")

    parser.add_argument("--max_num_labels", type=int, default=-1,
                        help="Use only use this number of labels for train/validation. Test set remains a fraction of the original total")
 
    parser.add_argument("--train_frac", type=float, default=0.9,
                        help="Fraction of non-test galaxies to use in the training set. Validation set receives the remaining")
 
    # parser.add_argument("--val_frac", type=float, default=0.1,
    #                     help="Fraction of galaxies to use in the validation set")
  
    parser.add_argument("--test_frac", type=float, default=0.1,
                        help="Fraction of galaxies to use in the test set")
         
    parser.add_argument("--min_votes", type=int, default=5,
                        help="Minimum number of votes required to use sample")

    parser.add_argument("--seed", type=int, default=13579,
                        help="Random seed")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")

    args = parser.parse_args()

    return args

def save_to_disk(inds, labels, num_votes, file_head):
	"""Loads in images and galaxy data from array of indices, and saves all to h5py file. 

	Also saves indices and labels in own .npy file
	"""

	print(f"saving {file_head}")
	np.save(f"{file_head}.npy", np.c_[inds, labels, num_votes])

	survey = 'south'
	to_rgb = False

	DDL = load_data.DecalsDataLoader(survey=survey, to_rgb=to_rgb)

	images = DDL.get_images(inds, npix_out=152)
	gals = DDL.get_fields(inds)

	with h5py.File(f"{file_head}.h5", "w") as f:
	    for k, v in gals.items():
	        if v.dtype=='float32' or v.dtype=='int32' and k != 'inds':
	            f.create_dataset(k, data=v)
	        
	    f.create_dataset('images', data=images)
	    f.create_dataset('num_votes', data=num_votes)
	    f.create_dataset('labels', data=labels)
	    f.create_dataset('inds', data=inds)

def main(args):

	params = vars(args) # convert args to dictionary

	np.random.seed(params['seed'])

	df_gz = pd.read_csv(params['data_path'])
	if params['verbose']:
		for col in df_gz.columns:
		    print(col)

	gz_question = 'smooth-or-featured'
	gz_label_col = gz_question+"_smooth_fraction"

	# gz_question = "how-rounded"
	# gz_label_col = gz_question+"_completely_fraction"

	# gz_question = "has-spiral-arms"         
	# gz_label_col = gz_question+"_yes"                                                                        

	gz_label_col_count = gz_question+"_total-votes"

	# Extract relevant columns from large Pandas dataframe
	labels = df_gz[gz_label_col].to_numpy()
	inds_decals = df_gz['index_in_decals'].to_numpy()
	num_votes = df_gz[gz_label_col_count].to_numpy()

	# Remove samples outside of desired range
	dm = (num_votes > params['min_votes']) & ~np.isnan(labels)
	labels = labels[dm]
	inds_decals = inds_decals[dm]
	num_votes = num_votes[dm]

	# Train/Val/Test split
	print(f"Total number of labels after cut: {len(labels)}")
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
	num_votes_train, num_votes_val, num_votes_test = num_votes[inds_train], num_votes[inds_val], num_votes[inds_test]
	inds_train, inds_val, inds_test = inds_decals[inds_train], inds_decals[inds_val], inds_decals[inds_test]

	save_to_disk(inds_train, labels_train, num_votes_train, f"../../data/morphology_labels_{gz_label_col}_train_{len(inds_train)}")
	save_to_disk(inds_val, labels_val, num_votes_val, f"../../data/morphology_labels_{gz_label_col}_val_{len(inds_val)}")
	# save_to_disk(inds_test, labels_test, num_votes_test, f"../../data/morphology_labels_{gz_label_col}_test_{len(inds_test)}")

if __name__ == '__main__':

	args = parse_arguments()

	main(args)
