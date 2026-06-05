from __future__ import annotations

import numpy as np

from stereo_spatial_calib.camera_geometry import depth_to_points_camera
from stereo_spatial_calib.coordinate_transform import transform_point
from stereo_spatial_calib.evaluator import estimate_object_center


def test_synthetic_depth_mask_center_identity_error_zero():
    intr = {"fx": 100.0, "fy": 100.0, "cx": 2.0, "cy": 2.0, "width": 5, "height": 5}
    depth = np.ones((5, 5), dtype=np.float32)
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    pts = depth_to_points_camera(depth, intr, mask=mask)
    center = estimate_object_center(pts, method="pointcloud_median")
    center_world = transform_point(np.eye(4), center)
    gt_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(center_world, gt_world, atol=1e-6)

