"""Isaac native RGB/depth capture routes.

This module is import-safe in a normal Python process. Isaac-specific imports happen
inside capture functions so synthetic tests do not depend on Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import traceback
from typing import Optional

import numpy as np

from .coordinate_transform import invert_transform, transform_point
from .io_utils import write_json


@dataclass
class NativeCaptureResult:
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    gt_depth_left: np.ndarray
    gt_instance_mask: Optional[np.ndarray]
    gt_segmentation: Optional[np.ndarray]
    metadata: dict
    capture_method: str
    mask_source: str


def as_numpy_frame(data):
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


def normalize_rgb(data):
    arr = as_numpy_frame(data)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim != 3:
        return None
    if arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        max_val = float(np.nanmax(arr)) if arr.size else 0.0
        if max_val <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def normalize_depth(data):
    arr = as_numpy_frame(data)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        return None
    arr = arr.astype(np.float32)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def camera_intrinsics_from_matrix(K, width, height):
    arr = np.asarray(K, dtype=np.float32)
    return {
        "fx": float(arr[0, 0]),
        "fy": float(arr[1, 1]),
        "cx": float(arr[0, 2]),
        "cy": float(arr[1, 2]),
        "width": int(width),
        "height": int(height),
    }


def default_intrinsics(width, height):
    fx = 600.0 * (float(width) / 640.0)
    return {
        "fx": float(fx),
        "fy": float(fx),
        "cx": (float(width) - 1.0) / 2.0,
        "cy": (float(height) - 1.0) / 2.0,
        "width": int(width),
        "height": int(height),
    }


def _project_points_cv(points_world, T_world_cam, intrinsics):
    T_cam_world = invert_transform(T_world_cam)
    pts_cam = np.asarray([transform_point(T_cam_world, p) for p in points_world], dtype=np.float32)
    z = pts_cam[:, 2]
    valid = np.isfinite(pts_cam).all(axis=1) & (z > 1e-6)
    uv = np.full((pts_cam.shape[0], 2), np.nan, dtype=np.float32)
    uv[valid, 0] = pts_cam[valid, 0] * float(intrinsics["fx"]) / z[valid] + float(intrinsics["cx"])
    uv[valid, 1] = pts_cam[valid, 1] * float(intrinsics["fy"]) / z[valid] + float(intrinsics["cy"])
    return uv, valid


def projected_bbox_mask_from_metadata(
    object_center_world,
    T_world_cam,
    intrinsics,
    object_type="cube",
    cube_size_m=0.08,
    sphere_radius_m=0.04,
    cylinder_radius_m=0.04,
    cylinder_height_m=0.08,
    shrink_ratio=0.8,
):
    """Generate a projected object mask from known pose and primitive size."""

    h = int(intrinsics["height"])
    w = int(intrinsics["width"])
    mask = np.zeros((h, w), dtype=bool)
    center = np.asarray(object_center_world, dtype=np.float32).reshape(3)
    if object_type == "cube":
        half = float(cube_size_m) / 2.0
        offsets = np.array(
            [[sx, sy, sz] for sx in (-half, half) for sy in (-half, half) for sz in (-half, half)],
            dtype=np.float32,
        )
        pts_world = center.reshape(1, 3) + offsets
        uv, valid = _project_points_cv(pts_world, T_world_cam, intrinsics)
        uv = uv[valid]
        if uv.size == 0:
            return mask
        x0, y0 = np.nanmin(uv, axis=0)
        x1, y1 = np.nanmax(uv, axis=0)
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        half_w = 0.5 * (x1 - x0) * float(shrink_ratio)
        half_h = 0.5 * (y1 - y0) * float(shrink_ratio)
        x0 = int(np.clip(np.floor(cx - half_w), 0, w - 1))
        x1 = int(np.clip(np.ceil(cx + half_w), 0, w - 1))
        y0 = int(np.clip(np.floor(cy - half_h), 0, h - 1))
        y1 = int(np.clip(np.ceil(cy + half_h), 0, h - 1))
        if x1 >= x0 and y1 >= y0:
            mask[y0 : y1 + 1, x0 : x1 + 1] = True
        return mask

    radius = cylinder_radius_m if object_type == "cylinder" else sphere_radius_m
    T_cam_world = invert_transform(T_world_cam)
    center_cam = transform_point(T_cam_world, center)
    if not np.isfinite(center_cam).all() or center_cam[2] <= 1e-6:
        return mask
    u = float(intrinsics["fx"]) * center_cam[0] / center_cam[2] + float(intrinsics["cx"])
    v = float(intrinsics["fy"]) * center_cam[1] / center_cam[2] + float(intrinsics["cy"])
    rx = max(2, int(round(float(intrinsics["fx"]) * radius / center_cam[2] * float(shrink_ratio))))
    if object_type == "cylinder":
        ry = max(2, int(round(float(intrinsics["fy"]) * (0.5 * cylinder_height_m) / center_cam[2] * shrink_ratio)))
    else:
        ry = max(2, int(round(float(intrinsics["fy"]) * radius / center_cam[2] * float(shrink_ratio))))
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - u) / max(rx, 1)) ** 2 + ((yy - v) / max(ry, 1)) ** 2) <= 1.0


def _step_render(app, count=1):
    async def _rep_step_async(rep):
        try:
            await rep.orchestrator.step_async(rt_subframes=4, delta_time=0.0, pause_timeline=False)
        except TypeError:
            try:
                await rep.orchestrator.step_async()
            except TypeError:
                await rep.orchestrator.step_async(rt_subframes=4)

    try:
        import omni.replicator.core as rep

        if hasattr(rep.orchestrator, "step_async"):
            _run_kit_coroutine(_rep_step_async(rep), app, max_updates=240)
        elif hasattr(rep.orchestrator, "step"):
            for _ in range(int(count)):
                try:
                    out = rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
                except TypeError:
                    out = rep.orchestrator.step()
                if inspect.isawaitable(out):
                    _run_kit_coroutine(out, app, max_updates=240)
                elif hasattr(rep.orchestrator, "wait_until_complete"):
                    rep.orchestrator.wait_until_complete()
    except Exception:
        pass
    for _ in range(max(1, int(count))):
        app.update()


def _run_kit_coroutine(coroutine, app, max_updates=240):
    """Run a Kit coroutine without taking over Kit's event loop."""

    from omni.kit.async_engine import run_coroutine

    task = run_coroutine(coroutine)
    for _ in range(int(max_updates)):
        app.update()
        if task.done():
            return task.result()
    raise TimeoutError("Timed out waiting for Kit coroutine.")


def _wait_render_product(app, render_product_path, frames=5):
    async def _wait():
        import omni.syntheticdata.sensors as sensors

        await sensors.next_render_simulation_async(str(render_product_path), int(frames))

    return _run_kit_coroutine(_wait(), app, max_updates=max(240, int(frames) + 180))


def _frame_valid(rgb, depth):
    if rgb is None or depth is None:
        return False
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.ndim != 2:
        return False
    return bool((np.isfinite(depth) & (depth > 0.0)).any())


def _frame_debug(rgb, depth, right_rgb=None):
    def shape_of(x):
        return None if x is None else list(np.asarray(x).shape)

    info = {
        "left_rgb_shape": shape_of(rgb),
        "right_rgb_shape": shape_of(right_rgb),
        "depth_shape": shape_of(depth),
    }
    if depth is not None:
        d = np.asarray(depth)
        valid = np.isfinite(d) & (d > 0.0)
        info.update(
            {
                "depth_valid_count": int(valid.sum()),
                "depth_valid_ratio": float(valid.sum() / valid.size) if d.size else 0.0,
                "depth_min": float(np.nanmin(d[valid])) if valid.any() else None,
                "depth_max": float(np.nanmax(d[valid])) if valid.any() else None,
            }
        )
    if rgb is not None:
        arr = np.asarray(rgb)
        info["left_rgb_mean"] = float(np.mean(arr)) if arr.size else None
        info["left_rgb_min"] = int(np.min(arr)) if arr.size else None
        info["left_rgb_max"] = int(np.max(arr)) if arr.size else None
    return info


def _camera_rgb(camera):
    for name in ("get_rgb", "get_rgba"):
        if hasattr(camera, name):
            rgb = normalize_rgb(getattr(camera, name)())
            if rgb is not None:
                return rgb
    try:
        frame = camera.get_current_frame(clone=True)
    except Exception:
        frame = {}
    for key in ("rgb", "rgba"):
        if key in frame:
            rgb = normalize_rgb(frame[key])
            if rgb is not None:
                return rgb
    return None


def _camera_depth(camera):
    candidates = []
    if hasattr(camera, "get_depth"):
        candidates.append(camera.get_depth())
    try:
        frame = camera.get_current_frame(clone=True)
    except Exception:
        frame = {}
    for key in ("distance_to_image_plane", "distance_to_camera", "depth"):
        if key in frame:
            candidates.append(frame[key])
    for raw in candidates:
        depth = normalize_depth(raw)
        if depth is not None and (np.isfinite(depth) & (depth > 0.0)).any():
            return depth
    for raw in candidates:
        depth = normalize_depth(raw)
        if depth is not None:
            return depth
    return None


def _camera_depth_from_pointcloud(camera, width, height):
    try:
        if hasattr(camera, "add_pointcloud_to_frame"):
            camera.add_pointcloud_to_frame()
    except Exception:
        pass
    try:
        cloud = as_numpy_frame(camera.get_pointcloud(world_frame=False))
    except Exception:
        return None
    if cloud is None:
        return None
    cloud = np.asarray(cloud, dtype=np.float32).reshape(-1, 3)
    if cloud.shape[0] != int(width) * int(height):
        return None
    depth = cloud[:, 2].reshape(int(height), int(width)).astype(np.float32)
    depth[~np.isfinite(depth)] = np.nan
    return depth


def _try_camera_mask(camera, shape):
    for add_name, key in (
        ("add_instance_segmentation_to_frame", "instance_segmentation"),
        ("add_instance_id_segmentation_to_frame", "instance_id_segmentation"),
        ("add_semantic_segmentation_to_frame", "semantic_segmentation"),
    ):
        try:
            if hasattr(camera, add_name):
                getattr(camera, add_name)()
            frame = camera.get_current_frame(clone=True)
            data = frame.get(key)
            raw = data.get("data") if isinstance(data, dict) else data
            arr = as_numpy_frame(raw)
            if arr is None:
                continue
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            if arr.shape[:2] == shape and (arr.astype(np.int64) > 0).any():
                return (arr.astype(np.int64) > 0), key
        except Exception:
            continue
    return None, None


def _base_metadata(sample_id, object_type, object_name, center_world, width, height, baseline_m, T_world_left, T_world_right):
    intr = default_intrinsics(width, height)
    return {
        "sample_id": sample_id,
        "backend": "isaac_native",
        "object_type": object_type,
        "object_name": object_name,
        "object_position_world": np.asarray(center_world, dtype=float).tolist(),
        "object_orientation_world": [0.0, 0.0, 0.0, 1.0],
        "object_center_world": np.asarray(center_world, dtype=float).tolist(),
        "left_camera_intrinsics": dict(intr),
        "right_camera_intrinsics": dict(intr),
        "baseline_m": float(baseline_m),
        "T_world_left_cam": np.asarray(T_world_left, dtype=np.float32).tolist(),
        "T_world_right_cam": np.asarray(T_world_right, dtype=np.float32).tolist(),
        "T_base_world": np.eye(4, dtype=np.float32).tolist(),
        "camera_frame_convention": "OpenCV/ROS optical: x right, y down, z forward",
    }


def _finish_result(
    sample_id,
    object_type,
    center_world,
    width,
    height,
    baseline_m,
    T_world_left,
    T_world_right,
    left_rgb,
    right_rgb,
    depth,
    mask,
    capture_method,
    mask_source,
    intr_left=None,
    intr_right=None,
    segmentation_available=False,
):
    metadata = _base_metadata(
        sample_id,
        object_type,
        f"target_{object_type}",
        center_world,
        width,
        height,
        baseline_m,
        T_world_left,
        T_world_right,
    )
    if intr_left is not None:
        metadata["left_camera_intrinsics"] = intr_left
    if intr_right is not None:
        metadata["right_camera_intrinsics"] = intr_right
    valid = np.isfinite(depth) & (depth > 0.0)
    metadata.update(
        {
            "native_capture_method": capture_method,
            "mask_source": mask_source,
            "segmentation_available": bool(segmentation_available),
            "rgb_shape": list(left_rgb.shape),
            "depth_shape": list(depth.shape),
            "depth_min_m": float(np.nanmin(depth[valid])) if valid.any() else None,
            "depth_max_m": float(np.nanmax(depth[valid])) if valid.any() else None,
            "depth_valid_ratio": float(valid.sum() / valid.size),
        }
    )
    return NativeCaptureResult(
        left_rgb=left_rgb,
        right_rgb=right_rgb,
        gt_depth_left=depth,
        gt_instance_mask=mask.astype(np.uint8) if mask is not None else None,
        gt_segmentation=mask.astype(np.int32) if mask is not None else None,
        metadata=metadata,
        capture_method=capture_method,
        mask_source=mask_source,
    )


def capture_with_camera_class(builder, sample_id, object_type, center_world, width, height, baseline_m):
    from isaacsim.sensors.camera import Camera

    builder.build_stage(object_type=object_type, object_position=np.asarray(center_world, dtype=float).tolist())
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
    for cam, pos, quat in ((left_cam, left_pos, left_quat), (right_cam, right_pos, right_quat)):
        if hasattr(cam, "set_world_pose"):
            cam.set_world_pose(position=pos, orientation=quat, camera_axes="ros")
        cam.initialize()
        if hasattr(cam, "set_resolution"):
            cam.set_resolution((int(width), int(height)))
        for add_name in ("add_rgb_to_frame", "add_distance_to_image_plane_to_frame", "add_distance_to_camera_to_frame"):
            if hasattr(cam, add_name):
                getattr(cam, add_name)()
        if hasattr(cam, "resume"):
            cam.resume()

    left_rgb = right_rgb = depth = None
    try:
        _wait_render_product(builder.simulation_app, left_cam.get_render_product_path(), frames=10)
        _wait_render_product(builder.simulation_app, right_cam.get_render_product_path(), frames=10)
    except Exception:
        _step_render(builder.simulation_app, count=10)
    for _ in range(80):
        _step_render(builder.simulation_app, count=1)
        try:
            _wait_render_product(builder.simulation_app, left_cam.get_render_product_path(), frames=1)
        except Exception:
            pass
        left_rgb = _camera_rgb(left_cam)
        right_rgb = _camera_rgb(right_cam)
        depth = _camera_depth(left_cam)
        if depth is None or not (np.isfinite(depth) & (depth > 0.0)).any():
            pc_depth = _camera_depth_from_pointcloud(left_cam, width, height)
            if pc_depth is not None:
                depth = pc_depth
        if _frame_valid(left_rgb, depth) and right_rgb is not None:
            break
    if not _frame_valid(left_rgb, depth) or right_rgb is None:
        raise RuntimeError(f"camera_class did not produce valid RGB/depth frames: {_frame_debug(left_rgb, depth, right_rgb)}")

    intr_left = intr_right = None
    if hasattr(left_cam, "get_intrinsics_matrix"):
        intr_left = camera_intrinsics_from_matrix(left_cam.get_intrinsics_matrix(), width, height)
    if hasattr(right_cam, "get_intrinsics_matrix"):
        intr_right = camera_intrinsics_from_matrix(right_cam.get_intrinsics_matrix(), width, height)

    mask, mask_key = _try_camera_mask(left_cam, depth.shape)
    seg_available = mask is not None
    mask_source = mask_key or "projected_bbox_fallback"
    if mask is None:
        mask = projected_bbox_mask_from_metadata(center_world, T_world_left, intr_left or default_intrinsics(width, height), object_type)
    return _finish_result(
        sample_id,
        object_type,
        center_world,
        width,
        height,
        baseline_m,
        T_world_left,
        T_world_right,
        left_rgb,
        right_rgb,
        depth,
        mask,
        "camera_class",
        mask_source,
        intr_left=intr_left,
        intr_right=intr_right,
        segmentation_available=seg_available,
    )


def _rep_get_data(annotator):
    try:
        return annotator.get_data()
    except Exception:
        return None


def _rep_mask_from_data(data, shape):
    if data is None:
        return None
    raw = data.get("data") if isinstance(data, dict) else data
    arr = as_numpy_frame(raw)
    if arr is None:
        return None
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.shape[:2] != shape:
        return None
    mask = arr.astype(np.int64) > 0
    return mask if mask.any() else None


def capture_with_replicator_annotators(builder, sample_id, object_type, center_world, width, height, baseline_m):
    import omni.replicator.core as rep

    builder.build_stage(object_type=object_type, object_position=np.asarray(center_world, dtype=float).tolist())
    T_world_left, T_world_right = builder.camera_pose()
    left_pos = np.asarray(T_world_left[:3, 3], dtype=float)
    right_pos = np.asarray(T_world_right[:3, 3], dtype=float)
    forward = np.asarray(T_world_left[:3, 2], dtype=float)
    left_cam = rep.create.camera(position=tuple(left_pos.tolist()), look_at=tuple((left_pos + forward).tolist()))
    right_cam = rep.create.camera(position=tuple(right_pos.tolist()), look_at=tuple((right_pos + forward).tolist()))
    left_rp = rep.create.render_product(left_cam, (int(width), int(height)))
    right_rp = rep.create.render_product(right_cam, (int(width), int(height)))

    rgb_l = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_r = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annotators = []
    for candidate in ("distance_to_image_plane", "distance_to_camera"):
        try:
            depth_annotators.append((candidate, rep.AnnotatorRegistry.get_annotator(candidate)))
        except Exception:
            pass
    seg_l = None
    for seg_name in ("instance_segmentation", "semantic_segmentation"):
        try:
            seg_l = rep.AnnotatorRegistry.get_annotator(seg_name)
            break
        except Exception:
            seg_l = None

    rgb_l.attach(left_rp)
    rgb_r.attach(right_rp)
    if not depth_annotators:
        raise RuntimeError("No supported Replicator depth annotator found.")
    for _, annotator in depth_annotators:
        annotator.attach(left_rp)
    if seg_l is not None:
        seg_l.attach(left_rp)

    left_rgb = right_rgb = depth = None
    last_debug = {}
    for _ in range(80):
        _step_render(builder.simulation_app, count=1)
        left_rgb = normalize_rgb(_rep_get_data(rgb_l))
        right_rgb = normalize_rgb(_rep_get_data(rgb_r))
        depth = None
        for _, annotator in depth_annotators:
            candidate_depth = normalize_depth(_rep_get_data(annotator))
            if candidate_depth is not None and (np.isfinite(candidate_depth) & (candidate_depth > 0.0)).any():
                depth = candidate_depth
                break
            if depth is None:
                depth = candidate_depth
        last_debug = _frame_debug(left_rgb, depth, right_rgb)
        if _frame_valid(left_rgb, depth) and right_rgb is not None:
            break
    if not _frame_valid(left_rgb, depth) or right_rgb is None:
        # Fall back to Isaac Sim's test utility, which wraps the same annotator API
        # but drives the Replicator async step in the runtime-supported way.
        try:
            from isaacsim.test.utils.image_capture import capture_annotator_data_async

            async def _capture_all():
                left = await capture_annotator_data_async("rgb", render_product=left_rp)
                right = await capture_annotator_data_async("rgb", render_product=right_rp)
                d = None
                for candidate_name, _ in depth_annotators:
                    maybe = await capture_annotator_data_async(candidate_name, render_product=left_rp)
                    maybe_depth = normalize_depth(maybe)
                    if maybe_depth is not None and (np.isfinite(maybe_depth) & (maybe_depth > 0.0)).any():
                        d = maybe
                        break
                    if d is None:
                        d = maybe
                return left, right, d

            raw_left, raw_right, raw_depth = _run_kit_coroutine(_capture_all(), builder.simulation_app, max_updates=480)
            left_rgb = normalize_rgb(raw_left)
            right_rgb = normalize_rgb(raw_right)
            depth = normalize_depth(raw_depth)
            last_debug = _frame_debug(left_rgb, depth, right_rgb)
        except Exception as exc:
            last_debug["capture_annotator_data_async_error"] = repr(exc)
    if not _frame_valid(left_rgb, depth) or right_rgb is None:
        raise RuntimeError(f"replicator_annotator did not produce valid RGB/depth frames: {last_debug}")

    mask = None
    seg_available = False
    mask_source = "projected_bbox_fallback"
    if seg_l is not None:
        mask = _rep_mask_from_data(_rep_get_data(seg_l), depth.shape)
        seg_available = mask is not None
        if seg_available:
            mask_source = "replicator_segmentation"
    intr = default_intrinsics(width, height)
    if mask is None:
        mask = projected_bbox_mask_from_metadata(center_world, T_world_left, intr, object_type)
    return _finish_result(
        sample_id,
        object_type,
        center_world,
        width,
        height,
        baseline_m,
        T_world_left,
        T_world_right,
        left_rgb,
        right_rgb,
        depth,
        mask,
        "replicator_annotator",
        mask_source,
        intr_left=intr,
        intr_right=intr,
        segmentation_available=seg_available,
    )


def capture_with_tiled_camera(builder, sample_id, object_type, center_world, width, height, baseline_m):
    try:
        from isaaclab.sensors import TiledCamera, TiledCameraCfg  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"tiled_camera import failed: {exc}") from exc
    raise RuntimeError("tiled_camera route is importable but not wired to an IsaacLab interactive scene yet.")


def capture_native_auto(builder, sample_id, object_type, center_world, width, height, baseline_m, native_capture_method="auto"):
    methods = (
        ["camera_class", "replicator_annotator", "tiled_camera"]
        if native_capture_method == "auto"
        else [native_capture_method]
    )
    failures = []
    for method in methods:
        try:
            if method == "camera_class":
                result = capture_with_camera_class(builder, sample_id, object_type, center_world, width, height, baseline_m)
            elif method == "replicator_annotator":
                result = capture_with_replicator_annotators(
                    builder, sample_id, object_type, center_world, width, height, baseline_m
                )
            elif method == "tiled_camera":
                result = capture_with_tiled_camera(builder, sample_id, object_type, center_world, width, height, baseline_m)
            else:
                raise ValueError(f"unknown native_capture_method: {method}")
            result.metadata["native_capture_failures"] = failures
            return result
        except Exception as exc:
            failures.append({"method": method, "error": repr(exc), "traceback": traceback.format_exc()})
    raise RuntimeError(f"all native capture methods failed: {failures}")


def save_native_capture_result(sample_dir, result: NativeCaptureResult):
    from .io_utils import save_image

    sample_dir.mkdir(parents=True, exist_ok=True)
    save_image(sample_dir / "left_rgb.png", result.left_rgb)
    save_image(sample_dir / "right_rgb.png", result.right_rgb)
    np.save(sample_dir / "gt_depth_left.npy", result.gt_depth_left.astype(np.float32))
    mask = result.gt_instance_mask
    if mask is None:
        mask = np.zeros_like(result.gt_depth_left, dtype=np.uint8)
    np.save(sample_dir / "gt_instance_mask.npy", mask.astype(np.uint8))
    seg = result.gt_segmentation if result.gt_segmentation is not None else mask.astype(np.int32)
    np.save(sample_dir / "gt_segmentation.npy", seg.astype(np.int32))
    write_json(sample_dir / "metadata.json", result.metadata)
