"""Dataset collection utilities.

This first implementation includes a deterministic synthetic rectified stereo backend
that mirrors the expected Isaac outputs. It is intentionally structured so an Isaac
collector can later replace only the capture layer while preserving metadata contracts.
"""

from __future__ import annotations

from pathlib import Path
import os
import traceback

import numpy as np

from .coordinate_transform import invert_transform, transform_point
from .io_utils import ensure_dir, save_image, write_json
from .isaac_scene_builder import IsaacStereoSceneBuilder
from .metadata_validator import assert_valid_sample_metadata
from .native_capture_methods import capture_native_auto, save_native_capture_result


def make_intrinsics(width, height, fx=None, fy=None):
    fx = float(fx if fx is not None else 600.0 * (width / 640.0))
    fy = float(fy if fy is not None else fx)
    return {
        "fx": fx,
        "fy": fy,
        "cx": (float(width) - 1.0) / 2.0,
        "cy": (float(height) - 1.0) / 2.0,
        "width": int(width),
        "height": int(height),
    }


def look_at_transform_cv(position_world, look_at_world, world_up=(0.0, 0.0, 1.0)):
    pos = np.asarray(position_world, dtype=np.float64)
    target = np.asarray(look_at_world, dtype=np.float64)
    forward = target - pos
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(world_up, dtype=np.float64)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = right.astype(np.float32)
    T[:3, 1] = down.astype(np.float32)
    T[:3, 2] = forward.astype(np.float32)
    T[:3, 3] = pos.astype(np.float32)
    return T


def project_point_cv(point_camera, intrinsics):
    x, y, z = [float(v) for v in point_camera]
    return (
        intrinsics["fx"] * x / z + intrinsics["cx"],
        intrinsics["fy"] * y / z + intrinsics["cy"],
    )


def _object_radius_pixels(object_type, depth_m, intrinsics):
    metric_radius = {"cube": 0.045, "sphere": 0.04, "cylinder": 0.04}.get(object_type, 0.04)
    rx = max(8, int(round(intrinsics["fx"] * metric_radius / depth_m)))
    ry = max(8, int(round(intrinsics["fy"] * metric_radius / depth_m)))
    if object_type == "cylinder":
        ry = int(ry * 1.35)
    return rx, ry


def _render_pair(object_type, center_cam, intrinsics, baseline_m):
    width = intrinsics["width"]
    height = intrinsics["height"]
    rgb_left = np.full((height, width, 3), [42, 46, 54], dtype=np.uint8)
    rgb_right = rgb_left.copy()
    gt_depth = np.full((height, width), 3.0, dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    seg = np.zeros((height, width), dtype=np.int32)

    # Simple table band for visual context.
    rgb_left[int(height * 0.62) :, :, :] = [92, 85, 73]
    rgb_right[int(height * 0.62) :, :, :] = [92, 85, 73]

    u, v = project_point_cv(center_cam, intrinsics)
    disparity = intrinsics["fx"] * baseline_m / center_cam[2]
    u_right = u - disparity
    rx, ry = _object_radius_pixels(object_type, center_cam[2], intrinsics)
    yy, xx = np.mgrid[0:height, 0:width]
    color = {
        "cube": np.array([220, 72, 64], dtype=np.uint8),
        "sphere": np.array([64, 170, 240], dtype=np.uint8),
        "cylinder": np.array([76, 205, 120], dtype=np.uint8),
    }.get(object_type, np.array([230, 230, 80], dtype=np.uint8))

    if object_type == "cube":
        mask = (np.abs(xx - u) <= rx) & (np.abs(yy - v) <= ry)
        mask_right = (np.abs(xx - u_right) <= rx) & (np.abs(yy - v) <= ry)
    else:
        mask = ((xx - u) / rx) ** 2 + ((yy - v) / ry) ** 2 <= 1.0
        mask_right = ((xx - u_right) / rx) ** 2 + ((yy - v) / ry) ** 2 <= 1.0
    rgb_left[mask] = color
    rgb_right[mask_right] = color
    gt_depth[mask] = float(center_cam[2])
    seg[mask] = 1
    return rgb_left, rgb_right, gt_depth, seg, mask


def generate_synthetic_stereo_dataset(
    output_dir,
    objects=("cube", "sphere", "cylinder"),
    num_samples_per_object=10,
    width=640,
    height=480,
    baseline_m=0.10,
    camera_position_world=(0.65, 0.0, 0.80),
    look_at_world=(0.45, 0.0, 0.05),
    seed=7,
):
    out = ensure_dir(output_dir)
    rng = np.random.default_rng(seed)
    intr = make_intrinsics(width, height)
    T_world_left = look_at_transform_cv(camera_position_world, look_at_world)
    T_left_world = invert_transform(T_world_left)
    right_offset_world = T_world_left[:3, 0] * float(baseline_m)
    T_world_right = np.array(T_world_left, copy=True)
    T_world_right[:3, 3] += right_offset_world
    T_base_world = np.eye(4, dtype=np.float32)

    created = []
    object_heights = {"cube": 0.04, "sphere": 0.04, "cylinder": 0.06}
    for object_type in objects:
        for idx in range(int(num_samples_per_object)):
            sample_id = f"{object_type}_{idx + 1:06d}"
            sample_dir = ensure_dir(out / sample_id)
            x = float(rng.uniform(0.35, 0.65))
            y = float(rng.uniform(-0.25, 0.25))
            z = float(object_heights.get(object_type, 0.04))
            center_world = np.array([x, y, z], dtype=np.float32)
            center_cam = transform_point(T_left_world, center_world)
            rgb_l, rgb_r, depth, seg, mask = _render_pair(object_type, center_cam, intr, float(baseline_m))
            save_image(sample_dir / "left_rgb.png", rgb_l)
            save_image(sample_dir / "right_rgb.png", rgb_r)
            np.save(sample_dir / "gt_depth_left.npy", depth)
            np.save(sample_dir / "gt_segmentation.npy", seg)
            np.save(sample_dir / "gt_instance_mask.npy", mask.astype(np.uint8))
            metadata = {
                "sample_id": sample_id,
                "backend": "synthetic_rectified_stereo",
                "object_type": object_type,
                "object_name": f"target_{object_type}",
                "object_position_world": center_world.tolist(),
                "object_orientation_world": [0.0, 0.0, 0.0, 1.0],
                "object_center_world": center_world.tolist(),
                "left_camera_intrinsics": intr,
                "right_camera_intrinsics": intr,
                "baseline_m": float(baseline_m),
                "T_world_left_cam": T_world_left.tolist(),
                "T_world_right_cam": T_world_right.tolist(),
                "T_base_world": T_base_world.tolist(),
                "instance_name_to_id": {f"target_{object_type}": 1},
                "camera_frame_convention": "OpenCV: x right, y down, z forward",
            }
            write_json(sample_dir / "metadata.json", metadata)
            created.append(sample_dir)
    return created


def generate_isaac_native_stereo_dataset(
    output_dir,
    reports_dir,
    objects=("cube", "sphere", "cylinder"),
    num_samples_per_object=10,
    width=640,
    height=480,
    baseline_m=0.10,
    camera_position_world=(0.5, -2.0, 0.50),
    look_at_world=(0.5, 0.0, 0.04),
    seed=7,
    native_capture_method="auto",
    mask_source="auto",
):
    """Collect a stereo dataset using Isaac Sim's native Camera sensor API."""

    log_path = ensure_dir(reports_dir) / "isaac_native_backend_error.log"
    report_path = ensure_dir(reports_dir) / "isaac_native_camera_capture_report.json"
    try:
        log_path.unlink()
    except FileNotFoundError:
        pass
    if os.environ.get("CONDA_DEFAULT_ENV") != "env_isaaclab":
        msg = (
            "isaac_native backend must run inside conda env_isaaclab. "
            "Run: source /home1/banghai/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab"
        )
        log_path.write_text(msg + "\n", encoding="utf-8")
        raise RuntimeError(msg)

    builder = None
    success = False
    try:
        builder = IsaacStereoSceneBuilder(
            {
                "width": width,
                "height": height,
                "baseline_m": baseline_m,
                "camera_position_world": camera_position_world,
                "look_at_world": look_at_world,
                "seed": seed,
                "native_capture_method": native_capture_method,
                "mask_source": mask_source,
            }
        )
        builder.launch()
        rng = np.random.default_rng(seed)
        out = ensure_dir(output_dir)
        created = []
        object_heights = {"cube": 0.04, "sphere": 0.04, "cylinder": 0.06}
        for object_type in objects:
            for idx in range(int(num_samples_per_object)):
                sample_id = f"{object_type}_{idx + 1:06d}"
                x = float(rng.uniform(0.35, 0.65))
                y = float(rng.uniform(-0.25, 0.25))
                z = float(object_heights.get(object_type, 0.04))
                sample_dir = ensure_dir(out / sample_id)
                result = capture_native_auto(
                    builder,
                    sample_id,
                    object_type,
                    np.array([x, y, z], dtype=np.float32),
                    width,
                    height,
                    baseline_m,
                    native_capture_method=native_capture_method,
                    mask_source=mask_source,
                )
                assert_valid_sample_metadata(
                    result.metadata,
                    rgb=result.left_rgb,
                    depth=result.gt_depth_left,
                    mask=result.gt_instance_mask,
                    expected_width=width,
                    expected_height=height,
                )
                save_native_capture_result(sample_dir, result)
                sample = sample_dir
                created.append(sample)
        if created:
            last_metadata = result.metadata
            write_json(
                report_path,
                {
                    "status": "success",
                    "sample_dir": str(created[0]),
                    "sample_count": len(created),
                    "native_capture_method": last_metadata.get("native_capture_method"),
                    "mask_source": last_metadata.get("mask_source"),
                    "segmentation_available": last_metadata.get("segmentation_available"),
                    "semantic_label_set_success": last_metadata.get("semantic_label_set_success"),
                    "target_instance_id": last_metadata.get("target_instance_id"),
                    "rgb_shape": last_metadata.get("rgb_shape"),
                    "depth_shape": last_metadata.get("depth_shape"),
                    "depth_min_m": last_metadata.get("depth_min_m"),
                    "depth_max_m": last_metadata.get("depth_max_m"),
                    "depth_valid_ratio": last_metadata.get("depth_valid_ratio"),
                },
            )
        success = True
        return created
    except Exception:
        tb = traceback.format_exc()
        log_path.write_text(tb, encoding="utf-8")
        write_json(
            report_path,
            {
                "status": "failed",
                "native_capture_method": native_capture_method,
                "error_log": str(log_path),
                "traceback_tail": tb[-4000:],
            },
        )
        raise
    finally:
        if (not success) and builder is not None:
            try:
                builder.close()
            except Exception:
                pass


def _camera_intrinsics_from_matrix(K, width, height):
    arr = np.asarray(K, dtype=np.float32)
    return {
        "fx": float(arr[0, 0]),
        "fy": float(arr[1, 1]),
        "cx": float(arr[0, 2]),
        "cy": float(arr[1, 2]),
        "width": int(width),
        "height": int(height),
    }


def _as_numpy_frame(data):
    if data is None:
        return None
    try:
        import warp as wp

        if isinstance(data, wp.array):
            data = data.numpy()
    except Exception:
        pass
    try:
        import torch

        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(data)
    if arr.size == 0:
        return None
    return arr


def _wait_for_camera_frame(app, camera, max_updates=24):
    """Wait for RGB/depth annotators using Isaac Sim's native render step API."""

    import omni.replicator.core as rep

    camera.resume()
    for _ in range(int(max_updates)):
        if hasattr(rep.orchestrator, "step"):
            rep.orchestrator.step()
        app.update()
        rgb = _as_numpy_frame(camera.get_rgb())
        depth = _as_numpy_frame(camera.get_depth())
        if depth is None:
            frame = camera.get_current_frame()
            depth = _as_numpy_frame(frame.get("distance_to_image_plane"))
        if rgb is not None and depth is not None and rgb.ndim == 3 and depth.ndim == 2:
            if rgb.shape[0] > 0 and rgb.shape[1] > 0 and depth.shape[0] > 0 and depth.shape[1] > 0:
                valid_depth = np.isfinite(depth) & (depth > 0.0)
                if valid_depth.any():
                    return rgb[:, :, :3].astype(np.uint8), depth.astype(np.float32)
    raise RuntimeError("Timed out waiting for valid Isaac Camera RGB/depth frame.")


def _projected_bbox_mask_from_pose(center_world, T_world_cam, intrinsics, object_type):
    T_cam_world = invert_transform(T_world_cam)
    center_cam = transform_point(T_cam_world, center_world)
    mask = np.zeros((intrinsics["height"], intrinsics["width"]), dtype=bool)
    if not np.isfinite(center_cam).all() or center_cam[2] <= 0:
        return mask
    u, v = project_point_cv(center_cam, intrinsics)
    rx, ry = _object_radius_pixels(object_type, center_cam[2], intrinsics)
    yy, xx = np.mgrid[0 : intrinsics["height"], 0 : intrinsics["width"]]
    if object_type == "cube":
        mask = (np.abs(xx - u) <= rx) & (np.abs(yy - v) <= ry)
    else:
        mask = ((xx - u) / rx) ** 2 + ((yy - v) / ry) ** 2 <= 1.0
    return mask


def _try_instance_mask(camera, shape):
    try:
        camera.add_instance_segmentation_to_frame()
        frame = camera.get_current_frame(clone=True)
        data = frame.get("instance_segmentation")
        if isinstance(data, dict):
            raw = data.get("data")
        else:
            raw = data
        arr = _as_numpy_frame(raw)
        if arr is None:
            return None
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.shape[:2] != shape:
            return None
        mask = arr.astype(np.int64) > 0
        if mask.any():
            return mask
    except Exception:
        return None
    return None


def _capture_isaac_native_sample(builder, sample_dir, sample_id, object_type, center_world, width, height, baseline_m):
    from isaacsim.sensors.camera import Camera

    stage = builder.build_stage(object_type=object_type, object_position=center_world.tolist())
    T_world_left, T_world_right = builder.camera_pose()
    left_pos, left_quat = builder.transform_to_ros_camera_pose(T_world_left)
    right_pos, right_quat = builder.transform_to_ros_camera_pose(T_world_right)

    left_cam = Camera(
        "/World/left_camera",
        name="left_camera",
        frequency=60,
        resolution=(int(width), int(height)),
        position=left_pos,
        orientation=left_quat,
    )
    right_cam = Camera(
        "/World/right_camera",
        name="right_camera",
        frequency=60,
        resolution=(int(width), int(height)),
        position=right_pos,
        orientation=right_quat,
    )
    left_cam.set_world_pose(position=left_pos, orientation=left_quat, camera_axes="ros")
    right_cam.set_world_pose(position=right_pos, orientation=right_quat, camera_axes="ros")
    left_cam.initialize()
    right_cam.initialize()
    left_cam.add_distance_to_image_plane_to_frame()
    right_cam.add_distance_to_image_plane_to_frame()
    left_cam.resume()
    right_cam.resume()
    for _ in range(10):
        builder.simulation_app.update()

    left_rgb, gt_depth = _wait_for_camera_frame(builder.simulation_app, left_cam)
    right_rgb, _ = _wait_for_camera_frame(builder.simulation_app, right_cam)
    K_left = left_cam.get_intrinsics_matrix()
    K_right = right_cam.get_intrinsics_matrix()
    intr_left = _camera_intrinsics_from_matrix(K_left, width, height)
    intr_right = _camera_intrinsics_from_matrix(K_right, width, height)

    mask = _try_instance_mask(left_cam, gt_depth.shape)
    mask_source = "native_instance"
    if mask is None:
        mask = _projected_bbox_mask_from_pose(center_world, T_world_left, intr_left, object_type)
        mask_source = "projected_bbox"

    save_image(sample_dir / "left_rgb.png", left_rgb)
    save_image(sample_dir / "right_rgb.png", right_rgb)
    np.save(sample_dir / "gt_depth_left.npy", gt_depth.astype(np.float32))
    np.save(sample_dir / "gt_instance_mask.npy", mask.astype(np.uint8))
    np.save(sample_dir / "gt_segmentation.npy", mask.astype(np.int32))

    metadata = {
        "sample_id": sample_id,
        "backend": "isaac_native",
        "object_type": object_type,
        "object_name": f"target_{object_type}",
        "object_position_world": center_world.astype(float).tolist(),
        "object_orientation_world": [0.0, 0.0, 0.0, 1.0],
        "object_center_world": center_world.astype(float).tolist(),
        "left_camera_intrinsics": intr_left,
        "right_camera_intrinsics": intr_right,
        "baseline_m": float(baseline_m),
        "T_world_left_cam": np.asarray(T_world_left, dtype=np.float32).tolist(),
        "T_world_right_cam": np.asarray(T_world_right, dtype=np.float32).tolist(),
        "T_base_world": np.eye(4, dtype=np.float32).tolist(),
        "mask_source": mask_source,
        "camera_frame_convention": "OpenCV/ROS optical: x right, y down, z forward",
    }
    write_json(sample_dir / "metadata.json", metadata)
    return sample_dir
