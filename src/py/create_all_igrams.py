"""
Top-level script to create all interferograms. Uses multiprocessing to speed things up.
"""

import multiprocessing
import os
import pathlib
import shutil
import sys
from methods.create_alos_interferogram import process_alos
from data.consts import ISCE2_OUTPUTS_DIR
from methods.igrams import SCHAEFER_INTEFEROGRAMS

igrams_to_do = SCHAEFER_INTEFEROGRAMS

# ensure no dups
assert len(set(igrams_to_do)) == len(igrams_to_do)

def worker(args):
    alos1, alos2 = args
    log_file = pathlib.Path(f"log_{alos1}_{alos2}.txt")
    # Redirect stdout to the log file
    fd = os.open(log_file, os.O_RDWR | os.O_CREAT)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    process_alos(alos1, alos2)

def main():
    print("STARTING PROCS...")
    num_processes = min(len(igrams_to_do), 24)
    with multiprocessing.Pool(num_processes) as pool:
        pool.map(worker, igrams_to_do)
    print("DONE")


if __name__ == '__main__':
    main()
