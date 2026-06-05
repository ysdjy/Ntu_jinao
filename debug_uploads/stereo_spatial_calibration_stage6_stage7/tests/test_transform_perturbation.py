from __future__ import annotations

import numpy as np

from stereo_spatial_calib.coordinate_transform import perturb_transform


def test_zero_transform_perturbation_preserves_identity():
    T = np.eye(4, dtype=np.float32)
    out = perturb_transform(T, trans_noise_std_m=0.0, rot_noise_deg=0.0, seed=123)
    np.testing.assert_allclose(out, T, atol=1e-7)


def test_transform_perturbation_reproducible_with_seed():
    T = np.eye(4, dtype=np.float32)
    a = perturb_transform(T, trans_noise_std_m=0.005, rot_noise_deg=0.5, seed=42)
    b = perturb_transform(T, trans_noise_std_m=0.005, rot_noise_deg=0.5, seed=42)
    c = perturb_transform(T, trans_noise_std_m=0.005, rot_noise_deg=0.5, seed=43)
    np.testing.assert_allclose(a, b, atol=1e-9)
    assert not np.allclose(a, c)

