"""
A helper to download files not from command line. This allows you to use the VSCode debugger.
"""

from pp.data.sar import _download_alos_palsar_granule

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
