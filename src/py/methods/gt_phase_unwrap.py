
import numpy as np

SEED = 10


def solve_best_phase_unwrap(gt_phases: np.ndarray, wrapped_phases: np.ndarray) -> np.ndarray:
    ks = []
    for (gt_p, w_p) in zip(gt_phases, wrapped_phases):
        k_ij = np.round((gt_p - w_p) / (2*np.pi))
        ks.append(k_ij)
    ks = np.array(ks)
    leftovers = gt_phases - 2 * np.pi * ks - wrapped_phases
    assert ((-np.pi <= leftovers) & (leftovers < np.pi)).all()
    global_offset = leftovers.mean()
    unwrapped_phase = wrapped_phases + ks * 2 * np.pi + global_offset
    return unwrapped_phase


def create_simulated_gt_phase(n, noise_per_pixel):
    mean = 30
    std = 2
    rand = np.random.RandomState(SEED)
    
    gt_phase = rand.randn(n) * std + mean
    rand_global_offset = rand.uniform(-10, 10)
    gt_unwrapped_phase = gt_phase + rand_global_offset
    
    if noise_per_pixel:
        mean_noise = 0.5
        std_noise = 0.1
        gt_unwrapped_phase = gt_unwrapped_phase + rand.randn(n) * std_noise + mean_noise
    
    ks = np.round(gt_unwrapped_phase / (2*np.pi))
    
    wrapped_phase = gt_unwrapped_phase - ks * 2 * np.pi
    return gt_phase, wrapped_phase


def test_simple():
    gt_phase, wrapped_phase = create_simulated_gt_phase(100, False)
    gt_phase_hat = solve_best_phase_unwrap(gt_phase, wrapped_phase)

    avg_abs_err = np.mean(np.abs(gt_phase_hat - gt_phase))
    print("simple phase err", avg_abs_err)
    
    
def test_complex():
    gt_phase, wrapped_phase = create_simulated_gt_phase(100, True)
    gt_phase_hat = solve_best_phase_unwrap(gt_phase, wrapped_phase)

    avg_abs_err = np.mean(np.abs(gt_phase_hat - gt_phase))
    print("complex phase err", avg_abs_err)


if __name__ == '__main__':
    test_simple()
    test_complex()

