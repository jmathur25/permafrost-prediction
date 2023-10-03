from methods.create_alos_interferogram import process_alos
from data.consts import ISCE2_OUTPUTS_DIR


igrams = [
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
    ("ALPSRP081662170", "ALPSRP189022170"),
    ("ALPSRP081662170", "ALPSRP242702170"),
    ("ALPSRP128632170", "ALPSRP182312170"),
    ("ALPSRP128632170", "ALPSRP189022170"),
    ("ALPSRP182312170", "ALPSRP189022170"),
    ("ALPSRP189022170", "ALPSRP235992170"),
    ("ALPSRP235992170", "ALPSRP242702170"),
]

for i in range(len(igrams)):
    for j in range(i + 1, len(igrams)):
        if igrams[i] == igrams[j]:
            print(f"DUP AT {i} {j}")

igrams_to_do = []
for alos1, alos2 in igrams:
    savedir = ISCE2_OUTPUTS_DIR / f"{alos1}_{alos2}"
    if savedir.exists():
        print("SKIPPING", alos1, alos2)
    else:
        igrams_to_do.append((alos1, alos2))

for alos1, alos2 in igrams_to_do:
    process_alos(alos1, alos2)
