"""Dataset metadata and array validators."""

from __future__ import annotations

import numpy as np


def _require(condition, message, errors):
    if not condition:
        errors.append(message)


def validate_sample_metadata(metadata, rgb=None, depth=None, mask=None, expected_width=640, expected_height=480):
    errors = []
    for key in (
        "sample_id",
        "object_type",
        "object_name",
        "object_center_world",
        "object_position_world",
        "object_orientation_world",
        "left_camera_intrinsics",
        "right_camera_intrinsics",
        "baseline_m",
        "T_world_left_cam",
        "T_world_right_cam",
        "T_base_world",
    ):
        _require(key in metadata, f"missing metadata key: {key}", errors)

    for side in ("left_camera_intrinsics", "right_camera_intrinsics"):
        intr = metadata.get(side, {})
        for key in ("fx", "fy", "cx", "cy", "width", "height"):
            _require(key in intr, f"missing {side}.{key}", errors)
        if intr:
            _require(float(intr.get("fx", 0.0)) > 0.0, f"{side}.fx must be > 0", errors)
            _require(float(intr.get("fy", 0.0)) > 0.0, f"{side}.fy must be > 0", errors)
            _require(int(intr.get("width", -1)) == int(expected_width), f"{side}.width mismatch", errors)
            _require(int(intr.get("height", -1)) == int(expected_height), f"{side}.height mismatch", errors)

    _require(float(metadata.get("baseline_m", 0.0)) > 0.0, "baseline_m must be > 0", errors)
    for key in ("T_world_left_cam", "T_world_right_cam", "T_base_world"):
        try:
            _require(np.asarray(metadata.get(key), dtype=np.float32).shape == (4, 4), f"{key} must be 4x4", errors)
        except Exception:
            errors.append(f"{key} must be numeric 4x4")
    _require(len(metadata.get("object_center_world", [])) == 3, "object_center_world must have length 3", errors)

    if rgb is not None:
        _require(np.asarray(rgb).shape == (int(expected_height), int(expected_width), 3), "rgb shape must be HxWx3", errors)
    if depth is not None:
        _require(np.asarray(depth).shape == (int(expected_height), int(expected_width)), "depth shape must be HxW", errors)
    if mask is not None and depth is not None:
        _require(np.asarray(mask).shape == np.asarray(depth).shape, "mask and depth shapes must match", errors)
    return errors


def assert_valid_sample_metadata(metadata, rgb=None, depth=None, mask=None, expected_width=640, expected_height=480):
    errors = validate_sample_metadata(
        metadata, rgb=rgb, depth=depth, mask=mask, expected_width=expected_width, expected_height=expected_height
    )
    if errors:
        raise ValueError("; ".join(errors))
    return True

