"""
Not relevant to paper. Sets up files for MintPy analysis. Might be outdated.
"""

import os
import pathlib

from data.consts import ALOS_L1_0_DIRNAME, ALOS_PALSAR_DATA_DIR, STACK_STRIPMAP_OUTPUTS_DIR

files = [f for f in os.listdir(ALOS_PALSAR_DATA_DIR) if f.startswith("ALPSR")]
OUTPUT_DIR = STACK_STRIPMAP_OUTPUTS_DIR / "barrow_2006_2010"

for f in files:
    src = pathlib.Path("..") / f / ALOS_L1_0_DIRNAME / "ARCHIVED_FILES" / f"{f}-L1.0.zip"
    src = src.absolute()
    dst = OUTPUT_DIR / "download" / src.name
    print(f"Symlinking {src} to {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)
    
    
