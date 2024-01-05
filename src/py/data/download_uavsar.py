# %%
# from uavsar_pytools.uavsar_tools import create_netrc
# create_netrc()

import os
from uavsar_pytools import UavsarScene

# %%
## Example url. Use vertex to find other urls: https://search.asf.alaska.edu/
zip_url = 'https://datapool.asf.alaska.edu/INTERFEROMETRY_GRD/UA/barrow_15018_17069-003_17098-001_0087d_s01_L090HH_02_int_grd.zip'

## Change this variable to a directory you want to download files into
image_directory = '/permafrost-prediction-shared-data/uavsar/barrow'

os.makedirs(image_directory, exist_ok=True)

# Instantiating an instance of the UavsarScene class and downloading all images
scene = UavsarScene(url = zip_url, work_dir= image_directory)

# %%
scene.url_to_tiffs()


# %%
