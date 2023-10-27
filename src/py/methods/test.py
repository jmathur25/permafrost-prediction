
import sys

sys.path.append('..')
from methods.soil_models import liu_deformation_from_alt


df = liu_deformation_from_alt(1.0)
print(df)
