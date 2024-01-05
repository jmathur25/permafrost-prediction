# permafrost-prediction
This contains code to reproduce the results in "A Modification to the Remotely Sensed Active Layer Thickness ReSALT Algorithm to have Self-Consistency".

## Prerequisities
1. You will need an EarthData login to download ALOS PALSAR images. Get one from https://urs.earthdata.nasa.gov/.
1. Make sure Docker is installed. Docker version 24.0.5 was used when this was developed. Also, a display was assumed in some parts of the container setup. If this is not the case for you, a few small changes might be needed.
1. This code is tested on a desktop with Intel x86 architecture and running Ubuntu 22.04.3 LTS.

Now, setup the dev environment. Follow the instructions in `dev_setup/README.md` for MintPy. All remaining instructions are meant to be run in the Docker container.

Note: it is a good idea to use a debugger when running Python scripts. They allow you to set breakpoints and investigate intermediate values. Scripts also easily break and using the debugger will save you time in fixing the script. Please make an issue/PR for any errors encountered. I develop by defining a `launch.json` in `.vscode` and using that to test scripts. I'll share a few examples with the instructions below.

## Seeing the algorithm work
To understand the algorithm, look at `src/pp/tests/resalt.py`. You can run this test with:
```
cd src
pytest pp/tests/resalt.py
```
This uses simulated data to show how SCReSALT outperforms ReSALT on soil models that don't have constant porosities.

Tests can be debugged with the following `launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug test",
            "type": "python",
            "request": "launch",
            "cwd": "${workspaceFolder}/src",
            "program": "/root/tools/mambaforge/bin/pytest",
            "args": [
                "pp/tests/resalt.py",
                // Optional, to run just one test:
                // "-k",
                // "test_liu_smm"
            ],
            "console": "integratedTerminal",
        }
    ]
}
```

## Reproducing paper results
To make the soil model figures, run `src/py/methods/soil_models_figures.py`. This is meant to run in VSCode's Jupyter Notebook integration inside the Docker container.

To reproduce the paper results, first download data:
```bash
cd src/py
python3 -m data.download calm --site Barrow
python3 -m data.download barrow_temperature 1995 2013
# The list of all granules (from table A1 in Schaefer et al. 2015)
# ALPSRP235992170
# ALPSRP182312170
# ALPSRP128632170
# ALPSRP189022170
# ALPSRP021272170
# ALPSRP242702170
# ALPSRP081662170
# ALPSRP027982170
# ALPSRP074952170
# It might be best to do these one at a time to make sure things work, or at least test one first.
python3 -m data.download alos_palsar_granule <EarthData username> <EarthData password> <ALOS PALSAR granules, seperated by spaces> 
```

To process the SAR imagery into interferograms, create a file at `~/.netrc` (TODO: test again):
```bash
machine	urs.earthdata.nasa.gov
    login	<EarthData username>
    password	<EarthData password>
```

Next, run:
```bash
cd src/py
python3 create_all_igrams.py
```

Finally:
```bash
cd src/py
# This file should be looked at and edited to make settings changes.
python3 -m methods.run_analysis
```

`run_analysis` can be debugged with the following `launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug",
            "type": "python",
            "request": "launch",
            "cwd": "${workspaceFolder}/src",
            "module": "pp.methods.run_analysis",
            "console": "integratedTerminal",
        }
    ]
}
```

To reproduce the extracted results from the ReSALT data product, open `src/py/benchmark_resalt_barrow_data_product.py`, follow the directions, and run the script.

## Download the same raw data
For reproducibility, I've stored the raw data (interferograms, temperature, CALM) in Zenodo. These can be downloaded from TODO and
placed into the right locations. See `share_data.md` for how this was done and where to place the download files.

Once this has been done, `run_analysis.py` should work.

## Other


TODO: better integrate this

To cleanup unused imports:
```
pip install autoflake
find src/py -type f -name "*.py" -exec autoflake --remove-all-unused-imports --in-place {} \;
```

