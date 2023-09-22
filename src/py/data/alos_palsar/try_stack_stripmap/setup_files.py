import os
import pathlib

files = [f for f in os.listdir("..") if f.startswith("ALPSR")]

print(files)
print(len(files))

#USE = ["ALPSRP021272170", "ALPSRP027982170", "ALPSRP128632170"] 

for f in files:  # files
    src = pathlib.Path("..") / f / "l1.0_data" / "ARCHIVED_FILES" / f"{f}-L1.0.zip"
    src = src.absolute()
    dst = pathlib.Path("download") / src.name
    print(f"Symlinking {src} to {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)
