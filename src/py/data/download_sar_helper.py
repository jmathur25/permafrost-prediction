
import os
import sys
sys.path.append("/permafrost-prediction/src/py")
from data.sar import _download_alos_palsar_granule

username = os.environ["USER"]
password = os.environ['PASSWORD']
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
_download_alos_palsar_granule(username, password, granules)
