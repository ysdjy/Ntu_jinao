from __future__ import annotations

import numpy as np

from stereo_spatial_calib.camera_geometry import disparity_to_depth, perturb_baseline, perturb_intrinsics, pixel_to_point_camera


def test_fx_increase_reduces_backprojected_x_magnitude():
    intr = {"fx": 100.0, "fy": 100.0, "cx": 2.0, "cy": 2.0, "width": 8, "height": 8}
    p_base = pixel_to_point_camera(6, 2, 2.0, intr)
    p_pert = pixel_to_point_camera(6, 2, 2.0, perturb_intrinsics(intr, fx_relative_error=0.10))
    assert abs(p_pert[0]) < abs(p_base[0])
    np.testing.assert_allclose(p_pert[2], p_base[2])


def test_baseline_relative_error_affects_disparity_to_depth():
    disparity = np.array([[20.0]], dtype=np.float32)
    baseline = 0.10
    base_depth = disparity_to_depth(disparity, fx=100.0, baseline_m=baseline)
    pert_depth = disparity_to_depth(disparity, fx=100.0, baseline_m=perturb_baseline(baseline, 0.10))
    assert pert_depth[0, 0] > base_depth[0, 0]
    np.testing.assert_allclose(pert_depth[0, 0], base_depth[0, 0] * 1.10, rtol=1e-6)

