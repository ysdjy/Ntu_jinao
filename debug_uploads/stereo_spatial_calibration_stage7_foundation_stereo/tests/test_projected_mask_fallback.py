from __future__ import annotations

import numpy as np

from stereo_spatial_calib.isaac_data_collector import look_at_transform_cv
from stereo_spatial_calib.native_capture_methods import projected_bbox_mask_from_metadata


def test_projected_cube_bbox_mask_is_nonempty_and_image_sized():
    intr = {"fx": 600.0, "fy": 600.0, "cx": 319.5, "cy": 239.5, "width": 640, "height": 480}
    T_world_cam = look_at_transform_cv([0.65, 0.0, 0.80], [0.45, 0.0, 0.05])
    mask = projected_bbox_mask_from_metadata([0.5, 0.0, 0.04], T_world_cam, intr, object_type="cube")
    assert mask.shape == (480, 640)
    assert mask.dtype == np.bool_
    assert mask.sum() > 0


def test_projected_sphere_mask_is_nonempty():
    intr = {"fx": 600.0, "fy": 600.0, "cx": 319.5, "cy": 239.5, "width": 640, "height": 480}
    T_world_cam = look_at_transform_cv([0.65, 0.0, 0.80], [0.45, 0.0, 0.05])
    mask = projected_bbox_mask_from_metadata([0.5, 0.0, 0.04], T_world_cam, intr, object_type="sphere")
    assert mask.shape == (480, 640)
    assert mask.sum() > 0

