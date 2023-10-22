

import os
import pathlib


granules = [
    'ALPSRP027982170',
    'ALPSRP026522170',
    'ALPSRP021272170',
    'ALPSRP020901410',
    'ALPSRP019812170',
    'ALPSRP018792170',
    'ALPSRP017401410',
    'ALPSRP017332170',
    'ALPSRP016671410'
]

l1_folders = [
    pathlib.Path(f"/permafrost-prediction-shared-data/alos-palasar/{s}") for s in granules
]

for l1_folder in l1_folders:
    l1_folder = l1_folder.absolute()
    fp = '/root/tools/mambaforge/share/isce2/stripmapStack/prepRawALOS.py' # /opt/isce2/src/isce2/contrib/stack/stripmapStack/prepRawALOS.py
    cmd = f"python3 {fp} -i {l1_folder}"
    res = os.system(cmd)
    assert res == 0, "prepRawALOS failed"
print('done')
