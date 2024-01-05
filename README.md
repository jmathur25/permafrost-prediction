# permafrost-prediction

You will need an EarthData login to download ALOS PALSAR images. Get one from https://urs.earthdata.nasa.gov/.

To get started, first setup the dev environment. Follow the instructions in `dev_setup/README.md` for MintPy. All following instructions are meant to be run in Docker container.

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

Note: it is a good idea to use a debugger when running Python scripts. They easily break and it will save you time if the script crashes after running for a while. Please make an issue/PR for any errors encountered. I develop by defining a `launch.json` in `.vscode` and using that to test module scripts. For example, `run_analysis` can be debugged with the following `launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug run_analysis.py",
            "type": "python",
            "request": "launch",
            "cwd": "${workspaceFolder}/src/py",
            "module": "methods.run_analysis",
            "console": "integratedTerminal",
        },
    ]
}
```


TODO: better integrate this

To cleanup unused imports:
```
pip install autoflake
find src/py -type f -name "*.py" -exec autoflake --remove-all-unused-imports --in-place {} \;
```

