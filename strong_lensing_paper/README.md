# ssl-legacysurvey
This directory contains public data from our strong lens finding campaign - *Mining for strong gravitational lenses with self-supervised learning - [arxiv.org/abs/2110.00023](https://arxiv.org/abs/2110.00023)* 

- data-usage.ipynb demonstrates how to load and visualize the data products, which include:
	- data/training_lenses.tsv - lenses used for training. 
	- data/new_lenses.tsv - our list of new lens candidates. 
	- data/network_predictions/ - our top 100,000 network predictions for each survey region, using both the linear classification and fine-tuned models.

- Full data, including images, can be found in h5py files, which data-usage.ipynb will download for you. Else find the files here [portal.nersc.gov/project/cusp/ssl\_galaxy\_surveys/strong\_lens\_data/
](https://portal.nersc.gov/project/cusp/ssl_galaxy_surveys/strong_lens_data/), or through the globus link on the main project Readme.

data-examples.ipynb requires a few libraries. Create a conda environment to install them as follows:

```
### Create conda environment for data-examples.ipynb

conda create --name ssl-legacysurvey python=3.8 matplotlib numpy ipykernel h5py pandas python-wget

conda activate ssl-legacysurvey

python -m ipykernel install --user --name ssl-legacysurvey --display-name ssl-legacysurvey
```


