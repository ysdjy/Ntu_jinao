"""Point-cloud reconstruction and serialization utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .camera_geometry import depth_to_points_camera
from .coordinate_transform import transform_points


def depth_to_pointcloud_camera(depth, intrinsics, mask=None):
    return depth_to_points_camera(depth, intrinsics, mask=mask)


def filter_valid_points(points, max_depth_m=None):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    if max_depth_m is not None:
        valid &= pts[:, 2] > 0.0
        valid &= pts[:, 2] <= float(max_depth_m)
    return pts[valid]


def remove_outliers_iqr(points, whisker=1.5):
    pts = filter_valid_points(points)
    if pts.shape[0] < 8:
        return pts
    q1 = np.percentile(pts, 25, axis=0)
    q3 = np.percentile(pts, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - float(whisker) * iqr
    upper = q3 + float(whisker) * iqr
    keep = ((pts >= lower) & (pts <= upper)).all(axis=1)
    return pts[keep]


def camera_to_world(points_camera, T_world_cam):
    return transform_points(T_world_cam, points_camera)


def world_to_base(points_world, T_base_world):
    return transform_points(T_base_world, points_world)


def save_pointcloud_npy(path, points):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(points, dtype=np.float32))


def save_pointcloud_ply(path, points, colors=None):
    """Save an ASCII PLY without requiring open3d."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = filter_valid_points(points)
    cols = None
    if colors is not None:
        cols = np.asarray(colors).reshape(-1, 3)
        if cols.shape[0] != pts.shape[0]:
            cols = None
        elif cols.dtype != np.uint8:
            cols = np.clip(cols, 0, 255).astype(np.uint8)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if cols is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if cols is None:
            for x, y, z in pts:
                f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
        else:
            for (x, y, z), (r, g, b) in zip(pts, cols):
                f.write(f"{x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)}\n")


def reconstruct_depth_products(depth, intrinsics, T_world_cam, T_base_world, mask=None):
    pts_cam = depth_to_pointcloud_camera(depth, intrinsics, mask=mask)
    pts_world = camera_to_world(pts_cam, T_world_cam)
    pts_base = world_to_base(pts_world, T_base_world)
    return pts_cam, pts_world, pts_base

