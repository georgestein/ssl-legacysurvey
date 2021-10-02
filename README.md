# ssl-legacysurvey
This directory contains public data and codes for self-supervised learning on the DESI legacy surveys.

## For the live interactive similarity search app please see ***[share.streamlit.io/georgest\
ein/galaxy_search](https://share.streamlit.io/georgestein/galaxy_search)***, with code available at [github.com/georgestein/galaxy_search](https://github.com/georgestein/galaxy_search)


Data can be found here [portal.nersc.gov/project/cusp/ssl_galaxy_surveys/strong_lens_data/
](https://portal.nersc.gov/project/cusp/ssl_galaxy_surveys/strong_lens_data/)


### Create conda environment for data-examples.ipynb
'''
conda create --name ssl-legacysurvey python=3.8 matplotlib numpy ipykernel h5py pandas

conda activate ssl-legacysurvey

python -m ipykernel install --user --name ssl-legacysurvey --display-name ssl-legacysurvey
'''


