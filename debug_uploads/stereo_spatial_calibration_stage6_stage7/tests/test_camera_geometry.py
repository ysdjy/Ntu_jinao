from __future__ import annotations

import numpy as np

from stereo_spatial_calib.camera_geometry import (
    depth_to_points_camera,
    disparity_to_depth,
    pixel_to_point_camera,
    project_points,
)


def test_disparity_to_depth_formula():
    disparity = np.array([[60.0, 30.0, 0.0]], dtype=np.float32)
    depth = disparity_to_depth(disparity, fx=600.0, baseline_m=0.10, max_depth_m=10.0)
    assert np.isclose(depth[0, 0], 1.0)
    assert np.isclose(depth[0, 1], 2.0)
    assert np.isnan(depth[0, 2])


def test_pixel_to_point_camera_backprojection():
    intr = {"fx": 100.0, "fy": 100.0, "cx": 2.0, "cy": 2.0, "width": 5, "height": 5}
    p = pixel_to_point_camera(3, 4, 2.0, intr)
    np.testing.assert_allclose(p, [0.02, 0.04, 2.0], atol=1e-6)


def test_depth_to_points_camera_shape_and_mask():
    intr = {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0, "width": 3, "height": 2}
    depth = np.ones((2, 3), dtype=np.float32)
    mask = np.array([[1, 0, 1], [0, 0, 1]], dtype=bool)
    pts = depth_to_points_camera(depth, intr, mask=mask)
    assert pts.shape == (3, 3)


def test_project_points_roundtrip():
    intr = {"fx": 100.0, "fy": 100.0, "cx": 2.0, "cy": 2.0, "width": 5, "height": 5}
    p = pixel_to_point_camera(3, 4, 2.0, intr)
    uv = project_points(p.reshape(1, 3), intr)[0]
    np.testing.assert_allclose(uv, [3.0, 4.0], atol=1e-6)

