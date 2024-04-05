"""
This is used to save space if interferograms are taking up too much space on your machine.
"""

import os
import shutil

from pp.methods.igrams import SCHAEFER_INTEFEROGRAMS
from pp.data.consts import ISCE2_OUTPUTS_DIR, ALOS_PALSAR_DATA_DIR

def delete_path(p):
    if (p.exists()):
        print("deleting", p)
        if os.path.isfile(p):
            os.remove(p)
        else:
            shutil.rmtree(p)

def delete_all():
    isce_delete = [
        ("coregisteredSlc/highBand",False),
        ("coregisteredSlc/lowBand",False),
        ("interferogram/highBand",False),
        ("interferogram/lowBand",False),
        ("offsets",False),
        ("coregisteredSlc",False),
        ("geometry/z.rdr.full",False),
        ("geometry/z.rdr.full.vrt",False),
        ("geometry/z.rdr.full.xml",False),
        ("*_slc", True),
        ("*_raw", True),
        ("interferogram/phsig.cor*", True),
        ("interferogram/topophase*", True),
        ("ionosphere", False)
    ]

    for (alos1, alos2) in SCHAEFER_INTEFEROGRAMS:
        d = ISCE2_OUTPUTS_DIR / f'{alos1}_{alos2}'
        for (u, is_glob) in isce_delete:
            if is_glob:
                # Use glob to find all matching paths
                for p in d.glob(u):
                    if p.exists():
                        delete_path(p)
            p = d / u
            delete_path(p)
                
                
    alos_palsas_delete_paths = [
        'l1.0_data/ARCHIVED_FILES'
    ]

    all_alos = os.listdir(ALOS_PALSAR_DATA_DIR)
    for alos in all_alos:
        for path in alos_palsas_delete_paths:
            p = os.path.join(ALOS_PALSAR_DATA_DIR / alos, path)
            delete_path(p)
            
if __name__ == '__main__':
    delete_all()
