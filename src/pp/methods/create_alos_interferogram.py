"""
Creates a single interferogram.
"""

import os
import pathlib
import click
import xml.etree.ElementTree as ET

from pp.data.consts import ALOS_L1_0_DIRNAME, ALOS_PALSAR_DATA_DIR, ISCE2_OUTPUTS_DIR

CURRENT_DIR = pathlib.Path(__file__).parent

ALOS_TEMPLATE_XML = CURRENT_DIR / "alos_template.xml"
STRIPMAP_APP_TEMPLATE_XML = CURRENT_DIR / "stripmapApp_template.xml"
DEM_FILE = CURRENT_DIR / "demLat_N70_N72_Lon_W158_W154.dem.wgs84"


@click.command()
@click.argument("alos1")
@click.argument("alos2")
def main(alos1, alos2):
    process_alos(alos1, alos2)


def process_alos(alos1, alos2):
    """
    Sets up and runs `stripmapApp.py` to make an interferogram.
    """

    alos1_imagefile, alos1_leaderfile = get_alos_imagefile_leaderfile(alos1)
    alos2_imagefile, alos2_leaderfile = get_alos_imagefile_leaderfile(alos2)

    output_dir = ISCE2_OUTPUTS_DIR / f"{alos1}_{alos2}"
    print(f"Results will be saved to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    alos1_output_dir = output_dir / alos1
    alos2_output_dir = output_dir / alos2
    alos1_xml = output_dir / f"alos_{alos1}.xml"
    alos2_xml = output_dir / f"alos_{alos2}.xml"
    stripmapApp_xml = output_dir / f"stripmapApp_{alos1}_{alos2}.xml"
    print(f"Writing XML configs: {alos1_xml} {alos2_xml} {stripmapApp_xml}")

    modify_alos_xml(ALOS_TEMPLATE_XML, alos1_xml, alos1_imagefile, alos1_leaderfile, alos1_output_dir)
    modify_alos_xml(ALOS_TEMPLATE_XML, alos2_xml, alos2_imagefile, alos2_leaderfile, alos2_output_dir)
    # due to a bug in DEM handling in ISCE source code, DEMs must be in the current directory. To work
    # around this, this code sets up simlinks.
    modify_stripmap_xml(STRIPMAP_APP_TEMPLATE_XML, stripmapApp_xml, alos1_xml, alos2_xml, DEM_FILE.name)
    print("Setting up DEM symlinks:")
    for f in os.listdir(DEM_FILE.parent):
        if f.startswith("dem"):
            src = (DEM_FILE.parent / f).absolute()
            dest = output_dir / f
            print(f"\tsymlinking {src} to {dest}")
            os.symlink(src, dest)

    ret = os.system(f"cd {output_dir} && stripmapApp.py {stripmapApp_xml.name} --steps")
    assert ret == 0


def get_alos_imagefile_leaderfile(alos):
    search_dir = ALOS_PALSAR_DATA_DIR / alos / ALOS_L1_0_DIRNAME
    files = set(os.listdir(search_dir))
    ignore = set(["ARCHIVED_FILES"])
    diff = files - ignore
    assert len(diff) == 1, f"Unable to parse data in {search_dir}"
    data_dir = search_dir / list(diff)[0] / alos
    files = os.listdir(data_dir)
    leaderfile = None
    imagefile = None
    for f in files:
        if f.startswith("LED-"):
            assert leaderfile is None
            leaderfile = f
        elif f.startswith("IMG-HH"):
            assert imagefile is None
            imagefile = f
    assert imagefile is not None, f"Could not find imagefile in: {files}"
    assert leaderfile is not None, f"Could not find leaderfile in: {files}"
    return data_dir / imagefile, data_dir / leaderfile


def modify_alos_xml(filename_in, filename_out, new_imagefile_value, new_leaderfile_value, output_folder):
    with open(filename_in, "r") as fp:
        lines = fp.readlines()
    new_lines = []
    mapping = {
        "{{IMAGEFILE}}": str(new_imagefile_value),
        "{{LEADERFILE}}": str(new_leaderfile_value),
        "{{OUTPUT_FOLDER}}": str(output_folder),
    }
    for i, l in enumerate(lines):
        wrote = False
        for k, v in mapping.items():
            if k in l:
                new_lines.append(l.replace(k, v))
                wrote = True
                break
        if not wrote:
            new_lines.append(l)
    with open(filename_out, "w") as fp:
        fp.writelines(new_lines)


def modify_stripmap_xml(filename_in, filename_out, new_reference_value, new_secondary_value, dem_filepath):
    tree = ET.parse(filename_in)
    root = tree.getroot()
    insar = root.find(".//component[@name='insar']")
    assert insar is not None
    # Find the 'reference' and 'secondary' components under 'insar'
    reference = insar.find(".//component[@name='reference']")
    secondary = insar.find(".//component[@name='secondary']")
    dem = insar.find("./property[@name='demFilename']/value")

    assert reference is not None
    assert secondary is not None
    assert dem is not None
    catalog_ref = reference.find("catalog")
    catalog_ref.text = str(new_reference_value)

    catalog_sec = secondary.find("catalog")
    catalog_sec.text = str(new_secondary_value)

    dem.text = str(dem_filepath)

    tree.write(filename_out)


if __name__ == "__main__":
    main()
