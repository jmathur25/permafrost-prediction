
import sys

sys.path.append('..')
from methods.soil_models import alt_to_surface_deformation


df = alt_to_surface_deformation(1.0)
print(df)
