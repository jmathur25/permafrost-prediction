
set -eux

bash run_unPackALOS

stackStripMap.py -W ionosphere -s SLC/ -d demLat_N70_N72_Lon_W158_W154.dem.wgs84 -t 10000 -b 100000 -a 7 -r 3 -u snaphu --useGPU

chmod +x run_files/run_*

time run_files/run_01_reference
time run_files/run_02_focus_split
time run_files/run_03_geo2rdr_coarseResamp
time run_files/run_04_refineSecondaryTiming
time run_files/run_05_invertMisreg
time run_files/run_06_fineResamp
time run_files/run_07_denseOffset
time run_files/run_08_invertDenseOffsets
time run_files/run_09_resampleOffset
time run_files/run_10_replaceOffsets
time run_files/run_11_fineResamp
time run_files/run_12_grid_baseline
time run_files/run_13_igram
time run_files/run_14_igramLowBand
time run_files/run_15_igramHighBand
time run_files/run_16_iono
