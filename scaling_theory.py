
from enum import Enum
import numpy as np
import sys


sys.path.append("/permafrost-prediction/src/py")
# from methods.soil_models import compute_alt_f_deformation, alt_to_surface_deformation
from methods.soil_models import LiuSMM


SEED = 7


class RefBiasDirection(Enum):
    NONE = 0
    LOWER_THAN_AVG = 1
    HIGHER_THAN_AVG = 2


def create_simulated_data(n, ref_bias_direction: RefBiasDirection):
    sqrt_ddt_earlier = np.sqrt(10)
    sqrt_ddt_later = np.sqrt(200)
    sqrt_ddt_ratio = sqrt_ddt_earlier / sqrt_ddt_later
    mean_alt_calib = 0.2
    std_alt_calib = 0.02
    rand = np.random.RandomState(SEED)
    
    alt2 = rand.randn(n) * std_alt_calib + mean_alt_calib
    alt1 = sqrt_ddt_ratio * alt2
    
    sub1 = np.array([alt_to_surface_deformation(alt) for alt in alt1])
    sub2 = np.array([alt_to_surface_deformation(alt) for alt in alt2])
    
    if ref_bias_direction == RefBiasDirection.NONE:
        ref_point = rand.randint(0, len(alt1))
    elif ref_bias_direction == RefBiasDirection.LOWER_THAN_AVG:
        sorted_indices = np.argsort(alt1)
        num_25_percent = len(alt1) // 4
        ref_point = rand.choice(sorted_indices[:-num_25_percent])
    elif ref_bias_direction == RefBiasDirection.HIGHER_THAN_AVG:
        sorted_indices = np.argsort(alt1)
        num_25_percent = len(alt1) // 4
        ref_point = rand.choice(sorted_indices[-num_25_percent:])
    else:
        raise ValueError()

    print("AVERAGE X", np.mean(alt1 - alt1[ref_point]))
    return sub2 - sub1, alt2[ref_point], sqrt_ddt_ratio, alt2, alt1


def estimate_alts(sub, ref_alt_later, sqrt_ddt_ratio, ref_bias_direction: RefBiasDirection):
    ref_alt_earlier = sqrt_ddt_ratio * ref_alt_later
    liu_smm = LiuSMM()
    # sub_earlier = alt_to_surface_deformation(ref_alt_earlier)
    sub_earlier = liu_smm.deformation_from_alt(ref_alt_earlier)
    sub_later = sub_earlier + sub
    # alt_later = np.array([compute_alt_f_deformation(sub) if sub > 1e-3 else np.nan for sub in sub_later])
    alt_later = np.array([liu_smm.deformation_from_alt(sub) if sub > 1e-3 else np.nan for sub in sub_later])
    if ref_bias_direction == RefBiasDirection.LOWER_THAN_AVG:
        alt_earlier = sqrt_ddt_ratio * alt_later
        # gamma = alt_earlier - ref_alt_earlier
        # print("AVG GAMMA", np.mean(gamma), np.mean(gamma > 0))
        # alt_earlier[gamma < 0] = ref_alt_earlier
    elif ref_bias_direction == RefBiasDirection.HIGHER_THAN_AVG:
        alt_earlier = sqrt_ddt_ratio * alt_later
        # alt_earlier[alt_earlier > ref_alt_earlier] = ref_alt_earlier
    elif ref_bias_direction == RefBiasDirection.NONE:
        alt_earlier = np.array([ref_alt_earlier] * len(alt_later))
    else:
        raise ValueError()
    return alt_later, alt_earlier


def test():
    ref_bias_direction = RefBiasDirection.LOWER_THAN_AVG
    sub, ref_alt_later, sqrt_ddt_ratio, alt2, alt1 = create_simulated_data(100, ref_bias_direction)
    alt_diff = alt2 - alt1
    assert (alt_diff > 0).all()
    
    # Solve using optimal scaling
    alt2_hat_opt, alt1_hat_opt = estimate_alts(sub, ref_alt_later, sqrt_ddt_ratio, ref_bias_direction)
    alt_diff_hat_opt = alt2_hat_opt - alt1_hat_opt
    assert (alt_diff_hat_opt > 0).all()
    print("REFERENCE-BIAS ACCOUNTED TOTAL ERR", np.mean(np.abs(alt_diff - alt_diff_hat_opt)))
    
    # Solve with no care for bias direction
    alt2_hat_subopt, alt1_hat_subopt = estimate_alts(sub, ref_alt_later, sqrt_ddt_ratio, RefBiasDirection.NONE)
    alt_diff_hat_subopt = alt2_hat_subopt - alt1_hat_subopt
    assert (alt_diff_hat_subopt > 0).all()
    print("NO BIAS TOTAL ERR", np.mean(np.abs(alt_diff - alt_diff_hat_subopt)))
    
    # ALT2 predictions should be the same
    assert np.isclose(alt2_hat_opt, alt2_hat_subopt).all()
    

if __name__ == '__main__':
    test()

