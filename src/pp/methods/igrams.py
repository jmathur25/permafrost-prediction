
# This comes from table A1 in Schaefer et al. (2015). It corrects a few errors in the table.
# Also, one of the interferograms had a processing error and was excluded.
SCHAEFER_INTEFEROGRAMS = [
    ("ALPSRP021272170", "ALPSRP027982170"),
    ("ALPSRP021272170", "ALPSRP128632170"),
    ("ALPSRP021272170", "ALPSRP182312170"),
    ("ALPSRP021272170", "ALPSRP189022170"),
    ("ALPSRP027982170", "ALPSRP182312170"),
    ("ALPSRP074952170", "ALPSRP081662170"),
    ("ALPSRP074952170", "ALPSRP128632170"),
    ("ALPSRP074952170", "ALPSRP182312170"),
    # ("ALPSRP074952170", "ALPSRP128632170"), # dup 6
    ("ALPSRP074952170", "ALPSRP189022170"),  # fix
    ("ALPSRP074952170", "ALPSRP235992170"),
    ("ALPSRP081662170", "ALPSRP128632170"),
    ("ALPSRP081662170", "ALPSRP182312170"),
    # ("ALPSRP081662170", "ALPSRP128632170"), # dup 10
    ("ALPSRP081662170", "ALPSRP189022170"),  # fix
    # ("ALPSRP081662170", "ALPSRP235992170"),  # wrong listing of palsar for '20100629' row 14. Code was not rerun with a fix. This is currently excluded.
    ("ALPSRP081662170", "ALPSRP242702170"),
    ("ALPSRP128632170", "ALPSRP182312170"),
    ("ALPSRP128632170", "ALPSRP189022170"),
    ("ALPSRP182312170", "ALPSRP189022170"),
    ("ALPSRP189022170", "ALPSRP235992170"),
    # ("ALPSRP235992170", "ALPSRP242702170"), # processing error
]

if __name__ == '__main__':
    all_granules = []
    for (g1, g2) in SCHAEFER_INTEFEROGRAMS:
        all_granules.append(g1)
        all_granules.append(g2)
    all_granules = set(all_granules)
    for granule in all_granules:
        print(granule)

