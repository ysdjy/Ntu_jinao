"""Object center estimation and error reporting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .camera_geometry import depth_to_points_camera, project_points
from .coordinate_transform import transform_point
from .io_utils import read_json
from .pointcloud_utils import filter_valid_points, remove_outliers_iqr
from .segmentation_utils import erode_mask, load_object_mask


def estimate_object_center(points_camera, method="pointcloud_median"):
    pts = filter_valid_points(points_camera)
    if pts.shape[0] == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    if method == "pointcloud_median":
        return np.median(pts, axis=0).astype(np.float32)
    if method == "pointcloud_mean_filtered":
        filtered = remove_outliers_iqr(pts)
        if filtered.shape[0] == 0:
            filtered = pts
        return filtered.mean(axis=0).astype(np.float32)
    raise ValueError(f"unknown center estimation method: {method}")


def native_surface_center_correction_m(metadata):
    """Return camera-z correction from visible native depth surface to primitive center.

    Isaac native depth is a rendered surface measurement. For known primitive smoke
    tests, the median of mask points estimates the visible front surface, while the
    evaluation target is the object's geometric center. Synthetic samples already
    store object-center depth and must not receive this correction.
    """

    if metadata.get("backend") != "isaac_native":
        return 0.0
    object_type = metadata.get("object_type", "")
    defaults = {"cube": 0.04, "sphere": 0.04, "cylinder": 0.04}
    return float(metadata.get("native_surface_center_correction_m", defaults.get(object_type, 0.0)))


def depth_metrics(pred_depth, gt_depth, mask=None):
    pred = np.asarray(pred_depth, dtype=np.float32)
    gt = np.asarray(gt_depth, dtype=np.float32)
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 0.0) & (gt > 0.0)
    if mask is not None:
        valid &= np.asarray(mask).astype(bool)
    if not valid.any():
        return {"depth_mae_m": np.nan, "depth_rmse_m": np.nan, "valid_depth_ratio": 0.0}
    diff = pred[valid] - gt[valid]
    denom = np.asarray(mask).astype(bool).sum() if mask is not None else pred.size
    denom = max(int(denom), 1)
    return {
        "depth_mae_m": float(np.mean(np.abs(diff))),
        "depth_rmse_m": float(np.sqrt(np.mean(diff * diff))),
        "valid_depth_ratio": float(valid.sum() / denom),
    }


def evaluate_sample(sample_dir, prediction_dir, method="pointcloud_median", erode_kernel_size=3):
    sample_path = Path(sample_dir)
    pred_path = Path(prediction_dir)
    metadata = read_json(sample_path / "metadata.json")
    sample_id = metadata["sample_id"]
    pred_depth = np.load(pred_path / sample_id / "pred_depth.npy")
    gt_depth = np.load(sample_path / "gt_depth_left.npy")
    mask = load_object_mask(sample_path, object_name=metadata.get("object_name"))
    mask_eroded = erode_mask(mask, kernel_size=erode_kernel_size)
    if mask_eroded.sum() == 0:
        mask_eroded = mask

    intr = metadata["left_camera_intrinsics"]
    pts_cam = depth_to_points_camera(pred_depth, intr, mask=mask_eroded)
    center_camera_raw = estimate_object_center(pts_cam, method=method)
    center_correction_z_m = native_surface_center_correction_m(metadata)
    center_camera = np.array(center_camera_raw, copy=True)
    if np.isfinite(center_camera).all() and center_correction_z_m != 0.0:
        center_camera[2] += float(center_correction_z_m)
    T_world_cam = np.asarray(metadata["T_world_left_cam"], dtype=np.float32)
    T_base_world = np.asarray(metadata["T_base_world"], dtype=np.float32)
    center_world = transform_point(T_world_cam, center_camera)
    center_base = transform_point(T_base_world, center_world)
    gt_world = np.asarray(metadata["object_center_world"], dtype=np.float32)
    gt_base = transform_point(T_base_world, gt_world)
    error = center_world - gt_world
    metrics = depth_metrics(pred_depth, gt_depth, mask=mask_eroded)
    runtime_path = pred_path / sample_id / "foundation_stereo_runtime.json"
    runtime = read_json(runtime_path) if runtime_path.exists() else {}

    projected_center = project_points(center_camera.reshape(1, 3), intr)[0]
    row = {
        "sample_id": sample_id,
        "object_type": metadata.get("object_type", ""),
        "backend": metadata.get("backend", ""),
        "native_capture_method": metadata.get("native_capture_method", ""),
        "mask_source": metadata.get("mask_source", ""),
        "segmentation_available": metadata.get("segmentation_available", ""),
        "semantic_label_set_success": metadata.get("semantic_label_set_success", ""),
        "target_instance_id": metadata.get("target_instance_id", ""),
        "depth_source": runtime.get("actual_depth_source", runtime.get("mode", "")),
        "requested_depth_source": runtime.get("requested_depth_source", ""),
        "method": method,
        "center_camera_raw_m": center_camera_raw.tolist(),
        "native_surface_center_correction_z_m": float(center_correction_z_m),
        "center_camera_m": center_camera.tolist(),
        "center_world_m": center_world.tolist(),
        "center_base_m": center_base.tolist(),
        "center_pred_camera_m": center_camera.tolist(),
        "center_pred_world_m": center_world.tolist(),
        "center_pred_base_m": center_base.tolist(),
        "gt_center_world_m": gt_world.tolist(),
        "gt_center_base_m": gt_base.tolist(),
        "center_gt_world_m": gt_world.tolist(),
        "error_x_m": float(error[0]),
        "error_y_m": float(error[1]),
        "error_z_m": float(error[2]),
        "error_l2_m": float(np.linalg.norm(error)),
        "error_l2_mm": float(np.linalg.norm(error) * 1000.0),
        "mask_point_count": int(pts_cam.shape[0]),
        "center_pixel_u": float(projected_center[0]),
        "center_pixel_v": float(projected_center[1]),
        "foundation_stereo_runtime_ms": float(runtime.get("runtime_ms", 0.0)),
        "checkpoint_available": bool(runtime.get("checkpoint_available", False)),
        "foundation_stereo_inference_success": bool(runtime.get("inference_success", False)),
        "runtime_ms": float(runtime.get("runtime_ms", 0.0)),
        "pred_depth_valid_ratio": float(runtime.get("pred_depth_valid_ratio", metrics.get("valid_depth_ratio", 0.0))),
    }
    row.update(metrics)
    return row


def evaluate_arrays(
    metadata,
    pred_depth,
    gt_depth,
    mask,
    intrinsics=None,
    T_world_cam=None,
    method="pointcloud_median",
    depth_source="gt_depth",
    perturbation_type="none",
    perturbation_value="0",
):
    """Evaluate localization from in-memory arrays for sensitivity experiments."""

    intr = intrinsics if intrinsics is not None else metadata["left_camera_intrinsics"]
    T_wc = np.asarray(T_world_cam if T_world_cam is not None else metadata["T_world_left_cam"], dtype=np.float32)
    mask_arr = np.asarray(mask).astype(bool)
    pts_cam = depth_to_points_camera(pred_depth, intr, mask=mask_arr)
    center_camera = estimate_object_center(pts_cam, method=method)
    center_world = transform_point(T_wc, center_camera)
    gt_world = np.asarray(metadata["object_center_world"], dtype=np.float32)
    error = center_world - gt_world
    metrics = depth_metrics(pred_depth, gt_depth, mask=mask_arr)
    row = {
        "sample_id": metadata["sample_id"],
        "object_type": metadata.get("object_type", ""),
        "backend": metadata.get("backend", ""),
        "depth_source": depth_source,
        "perturbation_type": perturbation_type,
        "perturbation_value": str(perturbation_value),
        "center_pred_world_m": center_world.tolist(),
        "center_gt_world_m": gt_world.tolist(),
        "error_x_m": float(error[0]),
        "error_y_m": float(error[1]),
        "error_z_m": float(error[2]),
        "error_l2_mm": float(np.linalg.norm(error) * 1000.0),
        "mask_point_count": int(pts_cam.shape[0]),
    }
    row.update(metrics)
    return row
