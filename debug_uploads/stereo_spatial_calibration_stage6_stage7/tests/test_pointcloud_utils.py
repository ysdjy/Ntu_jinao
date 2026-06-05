from __future__ import annotations

import numpy as np

from stereo_spatial_calib.pointcloud_utils import depth_to_pointcloud_camera, filter_valid_points, remove_outliers_iqr


def test_mask_extract_pointcloud():
    intr = {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0, "width": 3, "height": 3}
    depth = np.ones((3, 3), dtype=np.float32)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    pts = depth_to_pointcloud_camera(depth, intr, mask=mask)
    assert pts.shape == (1, 3)
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 1.0], atol=1e-6)


def test_filter_valid_points():
    pts = np.array([[0, 0, 1], [np.nan, 0, 1], [0, 0, 12]], dtype=np.float32)
    out = filter_valid_points(pts, max_depth_m=10.0)
    assert out.shape == (1, 3)


def test_remove_outliers_iqr_keeps_core_points():
    pts = np.array([[0, 0, 1], [0.01, 0, 1], [0, 0.01, 1], [0.01, 0.01, 1], [10, 10, 10]], dtype=np.float32)
    out = remove_outliers_iqr(pts)
    assert out.shape[0] >= 4

