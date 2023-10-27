import os
import shutil
from typing import Tuple
import click
import asf_search as asf
from data.consts import ALOS_L1_0_DIRNAME, ALOS_PALSAR_DATA_DIR
from data.utils import prompt_user
import re


@click.command()
@click.argument("username")
@click.argument("password")
@click.argument("granules", nargs=-1)
def alos_palsar_granule(username, password, granules: Tuple[str]):
    _download_alos_palsar_granule(username, password, granules)
    

def _download_alos_palsar_granule(username, password, granules: Tuple[str]):
    assert len(granules) > 0
    chosen_granules = []
    savedirs = []
    for granule in granules:
        savedir = ALOS_PALSAR_DATA_DIR / granule
        if savedir.exists():
            if prompt_user(f"Granule {granule} already downloaded to {savedir}. Redownload?"):
                chosen_granules.append(granule)
                savedirs.append(savedir)
        else:
            chosen_granules.append(granule)
            savedirs.append(savedir)
    del granules
    if len(chosen_granules) == 0:
        return

    print(f"Searching for granules: {chosen_granules}")
    results = asf.granule_search(chosen_granules)
    # TODO: make sure if search is valid
    print(f"Results of search: {results}")
    session = asf.ASFSession().auth_with_creds(username, password)
    ALOS_PALSAR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading granules...")
    results.download(path=ALOS_PALSAR_DATA_DIR, session=session)

    prefix = "ALPSRP"
    all_downloaded_files = set(
        [f for f in os.listdir(ALOS_PALSAR_DATA_DIR) if not os.path.isdir(ALOS_PALSAR_DATA_DIR / f)]
    )
    print("Moving downloads into respective folders")
    l1_folders = []
    for granule, savedir in zip(chosen_granules, savedirs):
        assert granule.startswith(prefix), f"Unexpected granule: {granule}"
        n1 = int(granule[len(prefix) : -4])
        n2 = int(granule[-4:])
        matched_files = []
        for f in all_downloaded_files:
            # (hack) use regex to find which files were downloaded for a particular granule
            match = re.match(f".*{n1}.*{n2}.*", f)
            if match is None:
                continue
            elif len(match.groups()) > 1:
                raise ValueError(f"Got an invalid match count for {f}")
            matched_files.append(f)
        if savedir.exists():
            print(f"Removing existing directory {savedir}")
            shutil.rmtree(savedir)
        savedir.mkdir(exist_ok=False)
        l1_0_file_savedir = savedir / ALOS_L1_0_DIRNAME
        other_savedir = savedir / "other"
        l1_0_file_savedir.mkdir(exist_ok=False)
        other_savedir.mkdir(exist_ok=False)
        found_l1_0_file = False
        for f in matched_files:
            if "L1.0" in f:
                found_l1_0_file = True
                shutil.move(str(ALOS_PALSAR_DATA_DIR / f), str(l1_0_file_savedir))
                l1_folders.append(l1_0_file_savedir)
            else:
                shutil.move(str(ALOS_PALSAR_DATA_DIR / f), str(other_savedir))
            all_downloaded_files.remove(f)
        assert found_l1_0_file, f"Found no raw data file (with L1.0 in its name) for {granule}"

    assert len(all_downloaded_files) == 0, f"Could not match these files: {all_downloaded_files}"
    
    l1_folders = [savedir / ALOS_L1_0_DIRNAME for savedir in savedirs]

    print("Extracting the raw (L1.0) data")
    for l1_folder in l1_folders:
        l1_folder = l1_folder.absolute()
        fp = '/root/tools/mambaforge/share/isce2/stripmapStack/prepRawALOS.py' # /opt/isce2/src/isce2/contrib/stack/stripmapStack/prepRawALOS.py
        cmd = f"python3 {fp} -i {l1_folder}"
        res = os.system(cmd)
        assert res == 0, "prepRawALOS failed"
