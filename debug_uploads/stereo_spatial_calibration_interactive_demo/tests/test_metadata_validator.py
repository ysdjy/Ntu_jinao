from __future__ import annotations

import numpy as np

from stereo_spatial_calib.metadata_validator import assert_valid_sample_metadata, validate_sample_metadata


def _metadata(width=640, height=480):
    intr = {"fx": 600.0, "fy": 600.0, "cx": 319.5, "cy": 239.5, "width": width, "height": height}
    return {
        "sample_id": "cube_000001",
        "object_type": "cube",
        "object_name": "target_cube",
        "object_center_world": [0.5, 0.0, 0.04],
        "object_position_world": [0.5, 0.0, 0.04],
        "object_orientation_world": [0.0, 0.0, 0.0, 1.0],
        "left_camera_intrinsics": intr,
        "right_camera_intrinsics": dict(intr),
        "baseline_m": 0.10,
        "T_world_left_cam": np.eye(4).tolist(),
        "T_world_right_cam": np.eye(4).tolist(),
        "T_base_world": np.eye(4).tolist(),
    }


def test_metadata_validator_accepts_valid_sample():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32)
    mask = np.ones((480, 640), dtype=np.uint8)
    assert assert_valid_sample_metadata(_metadata(), rgb=rgb, depth=depth, mask=mask)


def test_metadata_validator_rejects_bad_intrinsics():
    meta = _metadata()
    meta["left_camera_intrinsics"]["fx"] = 0.0
    errors = validate_sample_metadata(meta, expected_width=640, expected_height=480)
    assert any("fx" in err for err in errors)

