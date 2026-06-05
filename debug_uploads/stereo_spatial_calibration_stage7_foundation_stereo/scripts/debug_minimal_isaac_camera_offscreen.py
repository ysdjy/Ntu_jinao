#!/usr/bin/env python3
"""Minimal Isaac offscreen camera smoke test.

This script intentionally does not reuse the stereo pipeline. It verifies the
lowest-level requirement for the native backend: an Isaac Sim camera can render a
non-black RGB image and finite positive depth in headless/offscreen mode.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _save_rgb(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        cv2.imwrite(str(path), np.asarray(rgb)[:, :, ::-1])
        return
    except Exception:
        pass
    from PIL import Image

    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def _save_depth_color(path, depth):
    from stereo_spatial_calib.io_utils import colorize_scalar

    arr = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(arr) & (arr > 0.0)
    color = colorize_scalar(np.where(valid, arr, np.nan))
    _save_rgb(path, color)


def _as_numpy(data):
    if data is None:
        return None
    try:
        import torch

        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import warp as wp

        if isinstance(data, wp.array):
            data = data.numpy()
    except Exception:
        pass
    arr = np.asarray(data)
    if arr.size == 0:
        return None
    return arr


def _normalize_rgb(data):
    arr = _as_numpy(data)
    if arr is None or arr.ndim != 3 or arr.shape[-1] < 3:
        return None
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        max_val = float(np.nanmax(arr)) if arr.size else 0.0
        if max_val <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _normalize_depth(data):
    arr = _as_numpy(data)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None
    arr[~np.isfinite(arr)] = np.nan
    return arr


def _safe_frame(camera):
    for kwargs in ({"clone": True}, {}):
        try:
            return camera.get_current_frame(**kwargs)
        except TypeError:
            continue
    return {}


def _collect_frame(camera):
    frame = _safe_frame(camera)
    frame_keys = sorted(list(frame.keys())) if isinstance(frame, dict) else []

    rgb = None
    for name in ("get_rgb", "get_rgba"):
        if hasattr(camera, name):
            try:
                rgb = _normalize_rgb(getattr(camera, name)())
            except Exception:
                rgb = None
            if rgb is not None:
                break
    if rgb is None and isinstance(frame, dict):
        for key in ("rgb", "rgba"):
            if key in frame:
                rgb = _normalize_rgb(frame[key])
                if rgb is not None:
                    break

    depth = None
    if hasattr(camera, "get_depth"):
        try:
            depth = _normalize_depth(camera.get_depth())
        except Exception:
            depth = None
    if depth is None and isinstance(frame, dict):
        for key in ("distance_to_image_plane", "distance_to_camera", "depth"):
            if key in frame:
                depth = _normalize_depth(frame[key])
                if depth is not None:
                    break
    return frame_keys, rgb, depth


def _stats(report, rgb, depth):
    if rgb is not None:
        report.update(
            {
                "rgb_shape": list(rgb.shape),
                "rgb_mean": float(np.mean(rgb)),
                "rgb_min": int(np.min(rgb)),
                "rgb_max": int(np.max(rgb)),
            }
        )
    else:
        report.update({"rgb_shape": None, "rgb_mean": None, "rgb_min": None, "rgb_max": None})
    if depth is not None:
        valid = np.isfinite(depth) & (depth > 0.0)
        report.update(
            {
                "depth_shape": list(depth.shape),
                "depth_valid_ratio": float(valid.sum() / valid.size) if depth.size else 0.0,
                "depth_min": float(np.nanmin(depth[valid])) if valid.any() else None,
                "depth_max": float(np.nanmax(depth[valid])) if valid.any() else None,
            }
        )
    else:
        report.update(
            {
                "depth_shape": None,
                "depth_valid_ratio": 0.0,
                "depth_min": None,
                "depth_max": None,
            }
        )


def _quat_wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Minimal Isaac offscreen Camera RGB/depth smoke test.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no_headless", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true", default=False)
    parser.add_argument("--pose_mode", choices=["front", "pipeline"], default="front")
    args, _ = parser.parse_known_args()

    output_dir = PROJECT_ROOT / "outputs" / "debug_minimal_camera"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    report_path = reports_dir / "debug_minimal_isaac_camera_offscreen_report.json"
    stage5_report_path = reports_dir / "stage5_minimal_camera_debug_report.json"
    summary_path = reports_dir / "stage5_minimal_camera_debug_summary.md"

    headless = False if args.no_headless else bool(args.headless)
    report = {
        "status": "failed",
        "python": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "headless": headless,
        "enable_cameras_env": os.environ.get("ENABLE_CAMERAS"),
        "app_launcher_used": False,
        "simulation_app_config": {"headless": headless, "enable_cameras": True},
        "camera_frame_keys": [],
        "rgb_shape": None,
        "rgb_mean": None,
        "rgb_min": None,
        "rgb_max": None,
        "depth_shape": None,
        "depth_valid_ratio": 0.0,
        "depth_min": None,
        "depth_max": None,
        "error": "",
        "traceback": "",
    }
    simulation_app = None
    try:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=headless, enable_cameras=True)
        simulation_app = app_launcher.app
        report["app_launcher_used"] = True

        from isaacsim.core.utils.extensions import enable_extension

        for ext_name in ("isaacsim.sensors.camera", "omni.syntheticdata", "omni.replicator.core"):
            try:
                enable_extension(ext_name)
            except Exception:
                pass
        for _ in range(5):
            simulation_app.update()

        import omni.kit.app
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import VisualCuboid
        from isaacsim.core.utils.stage import create_new_stage
        from isaacsim.sensors.camera import Camera
        from pxr import UsdGeom, UsdLux

        create_new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        world.scene.add(
            VisualCuboid(
                prim_path="/World/red_cube",
                name="red_cube",
                position=np.array([0.0, 0.0, 0.5]),
                scale=np.array([0.4, 0.4, 0.4]),
                size=1.0,
                color=np.array([1.0, 0.0, 0.0]),
            )
        )
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(2500.0)

        if args.pose_mode == "pipeline":
            from stereo_spatial_calib.isaac_scene_builder import IsaacStereoSceneBuilder

            T_world_cam = IsaacStereoSceneBuilder(
                {
                    "camera_position_world": (0.65, 0.0, 0.80),
                    "look_at_world": (0.45, 0.0, 0.05),
                    "baseline_m": 0.10,
                }
            ).camera_pose()[0]
            cam_pos, cam_quat_wxyz = IsaacStereoSceneBuilder.transform_to_world_camera_pose(T_world_cam)
        else:
            # Isaac Camera's "world" axes convention is +X forward, +Z up. A +90
            # deg yaw rotates camera +X to world +Y, so y=-2 looks at origin.
            cam_pos = np.array([0.0, -2.0, 0.5], dtype=np.float32)
            cam_quat_wxyz = np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float32)
        report["pose_mode"] = args.pose_mode
        report["camera_position_world"] = np.asarray(cam_pos, dtype=float).tolist()
        report["camera_orientation_wxyz"] = np.asarray(cam_quat_wxyz, dtype=float).tolist()
        camera = Camera(
            prim_path="/World/debug_camera",
            name="debug_camera",
            frequency=60,
            resolution=(int(args.width), int(args.height)),
            position=cam_pos,
            orientation=cam_quat_wxyz,
        )
        if hasattr(camera, "set_world_pose"):
            camera.set_world_pose(position=cam_pos, orientation=cam_quat_wxyz, camera_axes="world")
        camera.initialize()
        if hasattr(camera, "set_resolution"):
            camera.set_resolution((int(args.width), int(args.height)))
        for add_name in ("add_rgb_to_frame", "add_distance_to_image_plane_to_frame", "add_distance_to_camera_to_frame"):
            if hasattr(camera, add_name):
                getattr(camera, add_name)()
        if hasattr(camera, "resume"):
            camera.resume()

        world.reset()
        rgb = depth = None
        frame_keys = []
        for _ in range(100):
            world.step(render=True)
            simulation_app.update()
            frame_keys, rgb, depth = _collect_frame(camera)
            valid_depth = depth is not None and bool((np.isfinite(depth) & (depth > 0.0)).any())
            valid_rgb = rgb is not None and float(np.mean(rgb)) > 0.0 and int(np.max(rgb)) > 0
            if valid_rgb and valid_depth:
                break

        report["camera_frame_keys"] = frame_keys
        _stats(report, rgb, depth)
        if rgb is not None:
            _save_rgb(output_dir / "rgb.png", rgb)
        if depth is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / "depth.npy", depth.astype(np.float32))
            _save_depth_color(output_dir / "depth_color.png", depth)

        rgb_ok = rgb is not None and float(report["rgb_mean"] or 0.0) > 0.0 and int(report["rgb_max"] or 0) > 0
        depth_ok = depth is not None and float(report["depth_valid_ratio"] or 0.0) > 0.0
        if not rgb_ok or not depth_ok:
            raise RuntimeError(
                "minimal camera produced invalid output: "
                f"rgb_ok={rgb_ok}, depth_ok={depth_ok}, frame_keys={frame_keys}"
            )
        report["status"] = "success"
        report["rgb_path"] = str(output_dir / "rgb.png")
        report["depth_path"] = str(output_dir / "depth.npy")
        report["depth_color_path"] = str(output_dir / "depth_color.png")
    except Exception as exc:
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        _write_json(report_path, report)
        _write_json(stage5_report_path, report)
        summary = [
            "# Stage 5 Minimal Camera Debug Summary",
            "",
            f"- AppLauncher available/used: {report.get('app_launcher_used')}",
            f"- ENABLE_CAMERAS env: {report.get('enable_cameras_env')}",
            f"- Minimal smoke status: {report.get('status')}",
            f"- RGB non-black: {bool((report.get('rgb_mean') or 0.0) > 0.0 and (report.get('rgb_max') or 0) > 0)}",
            f"- RGB stats: mean={report.get('rgb_mean')}, min={report.get('rgb_min')}, max={report.get('rgb_max')}",
            f"- Depth valid: {bool((report.get('depth_valid_ratio') or 0.0) > 0.0)}",
            f"- Depth stats: valid_ratio={report.get('depth_valid_ratio')}, min={report.get('depth_min')}, max={report.get('depth_max')}",
            "- native_capture_methods.py backfill: pending minimal smoke result",
            "- Official camera demo: not run in this script",
        ]
        if report.get("error"):
            summary.extend(["", "## Error", "", str(report.get("error")), "", "Traceback is in the JSON report."])
        summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                pass
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
