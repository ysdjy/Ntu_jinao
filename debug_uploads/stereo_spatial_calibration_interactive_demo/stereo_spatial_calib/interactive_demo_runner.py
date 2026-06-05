"""Button-triggered interactive capture and localization runner."""

from __future__ import annotations

import html
import shutil
import traceback
from pathlib import Path

import numpy as np

from .evaluator import evaluate_sample
from .foundation_stereo_adapter import FoundationStereoAdapter
from .io_utils import colorize_scalar, ensure_dir, save_image, write_json
from .isaac_scene_builder import IsaacStereoSceneBuilder
from .native_capture_methods import capture_with_camera_class
from .pointcloud_utils import camera_to_world, depth_to_pointcloud_camera, save_pointcloud_ply
from .segmentation_utils import erode_mask
from .stereo_rig_controller import StereoRigController
from .visualization import overlay_mask, write_sample_visualizations
from .interactive_visualization import add_result_markers


def _save_capture(sample_dir, result):
    ensure_dir(sample_dir)
    save_image(sample_dir / "left_rgb.png", result.left_rgb)
    save_image(sample_dir / "right_rgb.png", result.right_rgb)
    np.save(sample_dir / "gt_depth_left.npy", result.gt_depth_left.astype(np.float32))
    mask = result.gt_instance_mask if result.gt_instance_mask is not None else np.zeros(result.gt_depth_left.shape, dtype=np.uint8)
    np.save(sample_dir / "gt_instance_mask.npy", mask.astype(np.uint8))
    np.save(sample_dir / "gt_segmentation.npy", mask.astype(np.int32))
    write_json(sample_dir / "metadata.json", result.metadata)


def _copy_if_exists(src, dst):
    src = Path(src)
    if src.exists():
        ensure_dir(Path(dst).parent)
        shutil.copy2(src, dst)


def _write_interactive_report(out_dir, result):
    out = ensure_dir(out_dir)
    report_json = out / "report.json"
    write_json(report_json, result)
    lines = [
        "# Stereo Spatial Calibration Interactive Demo Report",
        "",
        f"- status: `{result.get('status')}`",
        f"- depth_source: `{result.get('depth_source')}`",
        f"- mask_source: `{result.get('mask_source')}`",
        f"- pointcloud_reference_view: `{result.get('pointcloud_reference_view')}`",
        f"- runtime_ms: `{result.get('runtime_ms')}`",
        f"- depth_rmse_m: `{result.get('depth_rmse_m')}`",
        f"- error_l2_mm: `{result.get('error_l2_mm')}`",
        "",
        "## Coordinates",
        "",
        f"- center_pred_left_camera_m: `{result.get('center_pred_left_camera_m')}`",
        f"- center_pred_world_m: `{result.get('center_pred_world_m')}`",
        f"- center_pred_base_m: `{result.get('center_pred_base_m')}`",
        f"- center_gt_world_m: `{result.get('center_gt_world_m')}`",
        "",
        "## Stereo Rig",
        "",
        f"- baseline_expected_m: `{result.get('baseline_expected_m')}`",
        f"- baseline_actual_m: `{result.get('baseline_actual_m')}`",
        f"- baseline_error_m: `{result.get('baseline_error_m')}`",
        f"- rotation_error_deg: `{result.get('rotation_error_deg')}`",
        "",
        "## Images",
        "",
    ]
    for name in (
        "left_rgb.png",
        "right_rgb.png",
        "disparity_color.png",
        "pred_depth_color.png",
        "gt_depth_color.png",
        "depth_error_color.png",
        "mask_overlay.png",
        "object_center_overlay.png",
    ):
        if (out / name).exists():
            lines += [f"![{name}]({out / name})", ""]
    md_path = out / "report.md"
    html_path = out / "report.html"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    body = ["<html><body><h1>Stereo Spatial Calibration Interactive Demo Report</h1>"]
    for line in lines:
        if line.startswith("!["):
            name = line.split("](")[0][2:]
            path = line.split("](")[1].rstrip(")")
            body.append(f"<h3>{html.escape(name)}</h3><img src='{html.escape(path)}' style='max-width:900px'>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            body.append(f"<p>{html.escape(line)}</p>")
    body.append("</body></html>")
    html_path.write_text("\n".join(body), encoding="utf-8")


def run_interactive_capture_and_infer(
    target_object="cube",
    baseline_m=0.10,
    resolution=(640, 480),
    output_dir=None,
    depth_source="foundation_stereo",
    mask_source="native_instance",
    use_existing_scene=True,
):
    """Capture current stereo RGB, run selected depth source, and evaluate.

    FoundationStereo predicts depth for the left reference view. GT depth and
    object pose are only used for evaluation metrics.
    """

    project_root = Path(__file__).resolve().parents[1]
    out = ensure_dir(output_dir or (project_root / "outputs" / "interactive_demo" / "latest"))
    sample_id = "interactive_latest"
    pred_root = ensure_dir(out / "predictions")
    viz_dir = ensure_dir(out / "visualizations")
    report = {"status": "failed", "traceback": ""}
    try:
        width, height = int(resolution[0]), int(resolution[1])
        object_center = np.array([0.5375286, 0.1986069, 0.04], dtype=float)
        config = {
            "baseline_m": float(baseline_m),
            "camera_position_world": (0.65, 0.0, 0.80),
            "look_at_world": (0.45, 0.0, 0.05),
        }
        builder = IsaacStereoSceneBuilder(config)
        builder.launch()
        rig = StereoRigController().create_rig(baseline_m=baseline_m, resolution=(width, height))
        capture = capture_with_camera_class(
            builder,
            sample_id,
            target_object,
            object_center,
            width,
            height,
            baseline_m,
            mask_source=mask_source,
        )
        rig_info = rig.validate_baseline_and_orientation()
        capture.metadata.update(rig_info)
        capture.metadata["pointcloud_reference_view"] = "left"
        capture.metadata["T_world_reference_camera"] = capture.metadata["T_world_left_cam"]
        sample_dir = out
        _save_capture(sample_dir, capture)

        adapter = FoundationStereoAdapter()
        pred_row = adapter.process_sample(sample_dir, pred_root, depth_source=depth_source, allow_fallback=False)
        if pred_row.get("skipped"):
            raise RuntimeError(f"depth source skipped: {pred_row.get('skip_reason')}")
        eval_row = evaluate_sample(sample_dir, pred_root, method="pointcloud_median")

        pred_dir = pred_root / sample_id
        write_sample_visualizations(sample_dir, pred_root, eval_row, viz_dir)
        mask = np.load(sample_dir / "gt_instance_mask.npy").astype(bool)
        pred_depth = np.load(pred_dir / "pred_depth.npy")
        gt_depth = np.load(sample_dir / "gt_depth_left.npy")
        intr = capture.metadata["left_camera_intrinsics"]
        T_world_cam = np.asarray(capture.metadata["T_world_left_cam"], dtype=np.float32)
        object_mask = erode_mask(mask, kernel_size=3)
        if object_mask.sum() == 0:
            object_mask = mask
        pts_left = depth_to_pointcloud_camera(pred_depth, intr, mask=object_mask)
        pts_world = camera_to_world(pts_left, T_world_cam)
        save_pointcloud_ply(out / "object_pointcloud_left_camera.ply", pts_left)
        save_pointcloud_ply(out / "object_pointcloud_world.ply", pts_world)

        _copy_if_exists(pred_dir / "disparity_color.png", out / "disparity_color.png")
        _copy_if_exists(pred_dir / "pred_depth_color.png", out / "pred_depth_color.png")
        _copy_if_exists(viz_dir / f"{sample_id}_object_center_overlay.png", out / "object_center_overlay.png")
        _copy_if_exists(viz_dir / f"{sample_id}_depth_error_color.png", out / "depth_error_color.png")
        save_image(out / "gt_depth_color.png", colorize_scalar(gt_depth))
        save_image(out / "mask_overlay.png", overlay_mask(capture.left_rgb, mask))
        save_image(out / "overview.png", overlay_mask(capture.left_rgb, mask, alpha=0.25))
        try:
            add_result_markers(eval_row["center_pred_world_m"], eval_row["center_gt_world_m"])
        except Exception:
            pass

        report = {
            "status": "success",
            "sample_id": sample_id,
            "target_object": target_object,
            "depth_source": eval_row.get("depth_source"),
            "requested_depth_source": depth_source,
            "mask_source": capture.mask_source,
            "pointcloud_reference_view": "left",
            "T_world_reference_camera": capture.metadata["T_world_left_cam"],
            "center_pred_left_camera_m": eval_row.get("center_pred_left_camera_m", eval_row.get("center_pred_camera_m")),
            "center_pred_camera_m": eval_row.get("center_pred_camera_m"),
            "center_pred_world_m": eval_row.get("center_pred_world_m"),
            "center_pred_base_m": eval_row.get("center_pred_base_m"),
            "center_gt_world_m": eval_row.get("center_gt_world_m"),
            "error_l2_mm": eval_row.get("error_l2_mm"),
            "depth_rmse_m": eval_row.get("depth_rmse_m"),
            "runtime_ms": eval_row.get("runtime_ms"),
            "pred_depth_valid_ratio": eval_row.get("pred_depth_valid_ratio"),
            "baseline_expected_m": rig_info["baseline_expected_m"],
            "baseline_actual_m": capture.metadata.get("baseline_actual_m", rig_info["baseline_actual_m"]),
            "baseline_error_m": capture.metadata.get("baseline_error_m", rig_info["baseline_error_m"]),
            "rotation_error_deg": rig_info["rotation_error_deg"],
            "left_rgb_path": str(out / "left_rgb.png"),
            "right_rgb_path": str(out / "right_rgb.png"),
            "report_html_path": str(out / "report.html"),
            "object_pointcloud_world_path": str(out / "object_pointcloud_world.ply"),
        }
        _write_interactive_report(out, report)
        return report
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        _write_interactive_report(out, report)
        raise
    finally:
        try:
            builder.simulation_app.close()
        except Exception:
            pass

