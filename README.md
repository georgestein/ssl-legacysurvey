## ssl-legacysurvey

![alt text](figures/ssl_umap.png)

ssl-legacysurvey allows for self-supervised learning (SSL), classification, regression, similarity search, and dimensionality reduction/anomaly detection tasks on large image datasets.

Focusing on 76 million galaxy images that I pulled from the [Dark Energy Spectroscopic Instrument (DESI) Legacy Survey](https://www.legacysurvey.org/) Data Release 9, this project works to release trained models, tools, and datasets. The codebase is more broadly applicable to any image dataset (just modify the dataloader and augmentations!).

Networks are trained using the [PyTorch-Lightning](https://www.pytorchlightning.ai/) framework, and utilizing components from the PyTorch lightning Bolts toolbox. 

## Current Products 
1. Interactive similarity search app: [share.streamlit.io/georgestein/galaxy_search](https://share.streamlit.io/georgestein/galaxy_search)
	* Paper @ [arXiv 2110.13151](https://arxiv.org/abs/2110.13151)
	* Code @ [github.com/georgestein/galaxy_search](https://github.com/georgestein/galaxy_search)

2. Data products from *Mining for strong gravitational lenses with self-supervised learning*
	* Paper @ [arxiv.org/abs/2110.00023](https://arxiv.org/abs/2110.00023)
	* Data in `strong_lensing_paper/` 

3. Dataset of 76 million galaxy images and all relevant codes (see below!)

4. Trained models 


## Getting Started

We start with a quick overview of the main functionalities (see `notebooks/tutorial` for a short tutorial, or see `scripts/` and `scripts/slurm/` for full distributed training scripts). Installation, data set access, and additional details follow.

* **Self-supervised learning** (see train_ssl.py)
	- Performs self-supervised learning using a specified CNN encoder architecture.
	- Architecture specified by --backbone parameter (e.g. 'resnet18', 'resnet50', 'resnet152', ...). Accepts any torchvision model.
	- Currently supports Momentum Contrast v2 (MoCov2) constrastive SSL. 

* **Classification & regression** (see finetune.py)
	- Trains classification or regression models from: 
		- scratch: `--backbone_weights random`
		- models pretrained on imagenet: `--backbone_weights imagenet`
		- From a checkpoint on disk (i.e. self-supervised model), where a classification head is added on to a backbone encoder: `--checkpoint_path <checkpoint_path>`
 
*  **Extract representations or predictions from model** (see predict.py)
	*  Loads in a trained model from a checkpoint, passes dataset through model, and saves each batch of outputs from each GPU in seperate file. Then compiles batches of into large chunks. This can be used for: 
		*  representation extraction: `--extract_representations`
		*  classification/regression: `--prediction_type <prediction_type>`  
                                                                  
* **Similarity search** (see similarity\_search\_nxn.py)
	* - Performs an N x N similarity search, where N is the number of representations. Used to  construct my interactive [Galaxy Search](https://share.streamlit.io/georgestein/galaxy_search) app. Requires Facebook's [Faiss](https://github.com/facebookresearch/faiss). 	  
* **Dimensionality reduction** (see anomaly.py)
	* - Performs dimensionality reduction through PCA or UMAP, saving transforms & embeddings to disk. 	  
 	  
ssl_legacysurvey/ contains SSL modules and dataloaders

## Installation 

Install the package reqirements with conda

`conda env create -f environment.yml`

Activate conda environment, `conda activate ssl-legacysurvey`, and install ssl-legacysurvey package in your python environment:

`pip install -e .`

## Data Access

The data set is large (20 TB), so we have set up a Globus endpoint to easily copy files to your local machine. We have also included a small toy datasample in this repo at `data/tiny_dataset.h5`. The Globus endpoint is (NEW AS OF April 26 2024):

[https://app.globus.org/file-manager?origin_id=9fb0fc0e-e760-11ec-9bd2-2d2219dcc1fa&origin_path=%2F](https://app.globus.org/file-manager?origin_id=59c818dc-8542-46d8-80d9-ab144669c7b6&origin_path=%2Fssl-legacysurvey%2F)

The directory is organized as follows, where image/catalogue information for each survey is split into chunks of 1,000,000 galaxies (sorted by decreasing z-band flux) and saved in hdf5 format:

	images/
		south/ : Images and galaxy catalogue information from DR9 south
			images_npix152_000000000_001000000.h5
			...
			images_npix152_061000000_062000000.h5
			
		north/ : Images and galaxy catalogue information from DR9 north
			images_npix152_000000000_001000000.h5
			...
			images_npix152_014000000_014174203.h5
			
		south-train/ : DR9 south images used for self-supervised training. 
		               (3,500,000 galaxies uniformly sampled in magnitude)
			images_npix152_000000000_003500000.h5

Images are provided as 152 x 152 pixel cutouts centered on each galaxy and are provided as 3 channels, which correspond to g, r, and z band, respectively, for an array size per file of (1,000,000, 3, 152, 152). 

Each file also contains relevant galaxy catalogue information from the DR9 sweep catalogues. [See here](https://www.legacysurvey.org/dr9/catalogs/) for a detailed escription of each:

	'brickid', 'dec', 'ebv', 'fiberflux', 'flux', 'fracin', 'images', 'inds', 'maskbits', 
	'nobs', 'objid', 'psfdepth', 'psfsize', 'ra', 'release', 'source_type', 'transmission', 
	'z_phot_l68', 'z_phot_l95', 'z_phot_mean', 'z_phot_median', 
	'z_phot_std', 'z_phot_u68', 'z_phot_u95', 'z_spec', 'z_training'
	
To load in the desired data fields simply use the DecalsDataLoader (`ssl_legacysurvey/utils/load_data.py`) provided:

	from ssl_legacysurvey.utils import load_data
	
    ngals = 8
    image_dir = '/global/cfs/projectdirs/cusp/LBL/decals_galaxy_survey/images/south/'

    DDL = import_data.DecalsDataLoader(verbose=True, image_dir=image_dir)

    # Choose <ngals> by random indices from [0, dataset_size)
    inds_use = np.random.choice(DDL.ngals_tot, ngals, replace=False)

    fields = ['images', 'ra', 'flux', 'source_type'] 

    galaxies = DDL.get_data(inds_use, fields=fields)
    for k in galaxies:
        print(f"{k} shape:", galaxies[k].shape)
 

## Trained Models

Self-supervised models are released as pytorch checkpoints, in order to see the training/hyperparameter setup used. They can be found at the Globus endpoint described above. 

Currently including:

* ResNet50
* ResNet34 & ResNet18 coming soon!

