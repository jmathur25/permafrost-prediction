
import os
import sys
sys.path.append("/permafrost-prediction/src/py")
from data.sar import _download_alos_palsar_granule

username = input("username: ")
password = input("password: ")
granules = [
    'ALPSRP027982170',
    'ALPSRP026522180',
    'ALPSRP021272170',
    'ALPSRP020901420',
    'ALPSRP019812180',
    'ALPSRP017332180',
    'ALPSRP016671420'
]
_download_alos_palsar_granule(username, password, granules)
