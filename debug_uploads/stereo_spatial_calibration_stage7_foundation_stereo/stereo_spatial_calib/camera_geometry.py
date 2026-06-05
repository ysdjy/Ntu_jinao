"""Camera geometry utilities.

Coordinate convention used by this package:
- OpenCV camera frame: x right, y down, z forward.
- Image pixels: u right, v down.

Isaac Sim camera conventions can differ depending on API/USD transform usage. Dataset
metadata must store ``T_world_left_cam`` as the transform from this OpenCV camera frame
to world. Any Isaac camera frame conversion must happen before writing that metadata.
"""

from __future__ import annotations

import numpy as np


def intrinsics_array(intrinsics: dict) -> tuple[float, float, float, float, int, int]:
    return (
        float(intrinsics["fx"]),
        float(intrinsics["fy"]),
        float(intrinsics["cx"]),
        float(intrinsics["cy"]),
        int(intrinsics["width"]),
        int(intrinsics["height"]),
    )


def disparity_to_depth(
    disparity,
    fx,
    baseline_m,
    min_disp=1e-6,
    max_depth_m=10.0,
):
    """Convert pixel disparity to metric depth with invalid values set to NaN."""

    disp = np.asarray(disparity, dtype=np.float32)
    depth = np.full(disp.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disp) & (disp > float(min_disp))
    depth[valid] = float(fx) * float(baseline_m) / disp[valid]
    depth[(depth <= 0.0) | (depth > float(max_depth_m)) | ~np.isfinite(depth)] = np.nan
    return depth


def pixel_to_point_camera(u, v, depth, intrinsics):
    """Back-project one pixel to the OpenCV camera frame."""

    fx, fy, cx, cy, _, _ = intrinsics_array(intrinsics)
    z = float(depth)
    if not np.isfinite(z) or z <= 0.0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def depth_to_points_camera(depth, intrinsics, mask=None):
    """Back-project a depth image to an ``(N, 3)`` camera-frame point array."""

    depth_arr = np.asarray(depth, dtype=np.float32)
    fx, fy, cx, cy, width, height = intrinsics_array(intrinsics)
    if depth_arr.shape != (height, width):
        raise ValueError(f"depth shape {depth_arr.shape} does not match intrinsics {(height, width)}")

    valid = np.isfinite(depth_arr) & (depth_arr > 0.0)
    if mask is not None:
        mask_arr = np.asarray(mask).astype(bool)
        if mask_arr.shape != depth_arr.shape:
            raise ValueError(f"mask shape {mask_arr.shape} does not match depth shape {depth_arr.shape}")
        valid &= mask_arr

    vv, uu = np.nonzero(valid)
    z = depth_arr[vv, uu]
    x = (uu.astype(np.float32) - cx) * z / fx
    y = (vv.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def project_points(points_camera, intrinsics):
    """Project camera-frame points to pixel coordinates.

    Returns ``(N, 2)`` float pixel coordinates. Points with invalid or non-positive z
    produce NaN pixels.
    """

    points = np.asarray(points_camera, dtype=np.float32).reshape(-1, 3)
    fx, fy, cx, cy, _, _ = intrinsics_array(intrinsics)
    pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float32)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)
    pixels[valid, 0] = points[valid, 0] * fx / points[valid, 2] + cx
    pixels[valid, 1] = points[valid, 1] * fy / points[valid, 2] + cy
    return pixels


def apply_depth_noise(depth, std_m=0.0, bias_m=0.0, seed=0):
    """Return a perturbed depth image without modifying the input."""

    depth_arr = np.asarray(depth, dtype=np.float32)
    out = depth_arr.copy()
    valid = np.isfinite(out) & (out > 0.0)
    if float(std_m) != 0.0:
        rng = np.random.default_rng(seed)
        out[valid] += rng.normal(0.0, float(std_m), size=int(valid.sum())).astype(np.float32)
    if float(bias_m) != 0.0:
        out[valid] += float(bias_m)
    out[valid] = np.maximum(out[valid], 1e-6)
    return out


def perturb_intrinsics(
    intrinsics,
    fx_relative_error=0.0,
    fy_relative_error=0.0,
    cx_shift_px=0.0,
    cy_shift_px=0.0,
):
    """Return a copy of intrinsics with controlled focal/principal-point errors."""

    out = dict(intrinsics)
    out["fx"] = float(out["fx"]) * (1.0 + float(fx_relative_error))
    out["fy"] = float(out["fy"]) * (1.0 + float(fy_relative_error))
    out["cx"] = float(out["cx"]) + float(cx_shift_px)
    out["cy"] = float(out["cy"]) + float(cy_shift_px)
    return out


def perturb_baseline(baseline_m, relative_error=0.0):
    return float(baseline_m) * (1.0 + float(relative_error))
