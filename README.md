## Try out the live interactive similarity search app @  ***[share.streamlit.io/georgestein/galaxy_search](https://share.streamlit.io/georgestein/galaxy_search)***
with code available at [github.com/georgestein/galaxy_search](https://github.com/georgestein/galaxy_search)


# ssl-legacysurvey
This directory contains public data from our work on self-supervised learning for the Dark Energy Spectroscopic Instrument (DESI) Legacy Imaging Surveys’ Data Release 9.

We are currently hosting new strong lens candidates found in *Mining for strong gravitational lenses with self-supervised learning - [arxiv.org/abs/2110.00023](https://arxiv.org/abs/2110.00023)* 


- data-usage.ipynb demonstrates how to load and visualize the data products, which include:
	- data/training_lenses.tsv - lenses used for training. 
	- data/new_lenses.tsv - our list of new lens candidates. 
	- data/network_predictions/ - our top 100,000 network predictions for each survey region, using both the linear classification and fine-tuned models.

- Full data, including images, can be found in h5py files, which data-usage.ipynb will download for you. Else find the files here [portal.nersc.gov/project/cusp/ssl\_galaxy\_surveys/strong\_lens\_data/
](https://portal.nersc.gov/project/cusp/ssl_galaxy_surveys/strong_lens_data/).

data-examples.ipynb requires a few libraries. Create a conda environment to install them as follows:

```
### Create conda environment for data-examples.ipynb

conda create --name ssl-legacysurvey python=3.8 matplotlib numpy ipykernel h5py pandas python-wget

conda activate ssl-legacysurvey

python -m ipykernel install --user --name ssl-legacysurvey --display-name ssl-legacysurvey
```


