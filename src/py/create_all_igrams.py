import multiprocessing
import os
import pathlib
import shutil
import sys
from methods.create_alos_interferogram import process_alos
from data.consts import ISCE2_OUTPUTS_DIR
from methods.igrams import JATIN_SINGLE_SEASON_2006_IGRAMS, SCHAEFER_INTEFEROGRAMS

igrams = JATIN_SINGLE_SEASON_2006_IGRAMS

# ensure no dups
assert len(set(igrams)) == len(igrams)

# TODO: parallelize
igrams_to_do = []
for alos1, alos2 in igrams:
    savedir = ISCE2_OUTPUTS_DIR / f"{alos1}_{alos2}"
    if savedir.exists():
        print("SKIPPING", alos1, alos2)
    else:
        igrams_to_do.append((alos1, alos2))


def worker(args):
    alos1, alos2 = args
    log_file = pathlib.Path(f"log_{alos1}_{alos2}.txt")
    # Redirect stdout to the log file
    fd = os.open(log_file, os.O_RDWR | os.O_CREAT)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    process_alos(alos1, alos2)
    
print("STARTING PROCS...")
num_processes = min(len(igrams_to_do), 24)
with multiprocessing.Pool(num_processes) as pool:
    pool.map(worker, igrams_to_do)
print("DONE")
