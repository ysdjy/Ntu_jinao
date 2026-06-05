#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_spatial_calib.foundation_stereo_adapter import FoundationStereoAdapter  # noqa: E402
from stereo_spatial_calib.io_utils import OUTPUT_ROOT, colorize_scalar, load_image_rgb, read_json, save_image, write_json  # noqa: E402
from stereo_spatial_calib.isaac_data_collector import generate_isaac_native_stereo_dataset  # noqa: E402
from stereo_spatial_calib.segmentation_utils import load_object_mask  # noqa: E402
from stereo_spatial_calib.visualization import overlay_mask  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--baseline", type=float, default=0.10)
    parser.add_argument("--mask_source", default="auto")
    parser.add_argument("--repo_path", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--foundation_stereo_repo", default=None)
    parser.add_argument("--foundation_stereo_model_dir", default=None)
    parser.add_argument("--foundation_stereo_ckpt", default=None)
    parser.add_argument("--foundation_stereo_cfg", default=None)
    args = parser.parse_args()

    reports = OUTPUT_ROOT / "reports"
    pred_root = OUTPUT_ROOT / "predictions"
    viz = OUTPUT_ROOT / "visualizations"
    report_path = reports / "foundation_stereo_native_pair_report.json"
    dataset_dir = OUTPUT_ROOT / "foundation_stereo_native_pair_dataset"

    samples = generate_isaac_native_stereo_dataset(
        dataset_dir,
        reports_dir=reports,
        objects=["cube"],
        num_samples_per_object=1,
        width=args.resolution[0],
        height=args.resolution[1],
        baseline_m=args.baseline,
        seed=42,
        native_capture_method="camera_class",
        mask_source=args.mask_source,
    )
    sample = samples[0]
    metadata = read_json(sample / "metadata.json")
    adapter = FoundationStereoAdapter(
        repo_path=args.foundation_stereo_repo or args.repo_path,
        checkpoint_path=args.foundation_stereo_ckpt or args.checkpoint_path,
        model_dir=args.foundation_stereo_model_dir,
        cfg_path=args.foundation_stereo_cfg,
    )
    status = adapter.status()
    out_dir = pred_root / "foundation_stereo_native_pair"
    mask = load_object_mask(sample)
    left_rgb = load_image_rgb(sample / "left_rgb.png")
    right_rgb = load_image_rgb(sample / "right_rgb.png")
    gt_depth = np.load(sample / "gt_depth_left.npy")
    save_image(viz / "foundation_stereo_native_pair_left_rgb.png", left_rgb)
    save_image(viz / "foundation_stereo_native_pair_right_rgb.png", right_rgb)
    save_image(viz / "foundation_stereo_native_pair_gt_depth.png", colorize_scalar(gt_depth))
    save_image(viz / "foundation_stereo_native_pair_mask_overlay.png", overlay_mask(left_rgb, mask))
    if not adapter.ready:
        report = {
            "status": "checkpoint_missing",
            "checkpoint_ready": False,
            "checkpoint_missing": True,
            "inference_success": False,
            "runtime_ms": 0.0,
            "sample_dir": str(sample),
            "checkpoint_status": status,
            "input_left_path": str(sample / "left_rgb.png"),
            "input_right_path": str(sample / "right_rgb.png"),
            "disparity_shape": None,
            "pred_depth_shape": None,
            "pred_depth_valid_ratio": 0.0,
            "depth_mae_m": None,
            "depth_rmse_m": None,
            "depth_error_median_m": None,
            "depth_error_p90_m": None,
            "depth_error_p95_m": None,
            "mask_source": metadata.get("mask_source"),
            "mask_point_count": int(mask.sum()),
            "error": "checkpoint_missing",
            "traceback": "",
            "hint": "Place model_best_bp2.pth and cfg.yaml under third_party/FoundationStereo/pretrained_models/23-51-11/.",
        }
        write_json(report_path, report)
        print(json.dumps(report, indent=2))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    try:
        row = adapter.process_sample(sample, pred_root, depth_source="foundation_stereo", allow_fallback=False)
    except Exception as exc:
        report = {
            "status": "failed",
            "checkpoint_ready": bool(adapter.ready),
            "checkpoint_missing": False,
            "inference_success": False,
            "runtime_ms": 0.0,
            "sample_dir": str(sample),
            "checkpoint_status": status,
            "input_left_path": str(sample / "left_rgb.png"),
            "input_right_path": str(sample / "right_rgb.png"),
            "mask_source": metadata.get("mask_source"),
            "mask_point_count": int(mask.sum()),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(report_path, report)
        print(json.dumps(report, indent=2))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sample_pred = pred_root / metadata["sample_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("disparity.npy", "pred_depth.npy"):
        data = np.load(sample_pred / name)
        np.save(out_dir / name, data)
    disp = np.load(out_dir / "disparity.npy")
    pred_depth = np.load(out_dir / "pred_depth.npy")
    save_image(viz / "foundation_stereo_native_pair_disparity.png", colorize_scalar(disp))
    save_image(viz / "foundation_stereo_native_pair_pred_depth.png", colorize_scalar(pred_depth))
    save_image(viz / "foundation_stereo_native_pair_depth_error.png", colorize_scalar(np.abs(pred_depth - gt_depth)))
    valid = np.isfinite(pred_depth) & np.isfinite(gt_depth) & (pred_depth > 0.0) & (gt_depth > 0.0)
    diff_abs = np.abs(pred_depth[valid] - gt_depth[valid]) if valid.any() else np.array([], dtype=np.float32)
    report = {
        "status": "success",
        "checkpoint_ready": True,
        "checkpoint_missing": False,
        "inference_success": True,
        "sample_dir": str(sample),
        "prediction_dir": str(out_dir),
        "checkpoint_status": status,
        "runtime_ms": row.get("foundation_stereo_runtime_ms", 0.0),
        "input_left_path": str(sample / "left_rgb.png"),
        "input_right_path": str(sample / "right_rgb.png"),
        "disparity_shape": list(disp.shape),
        "pred_depth_shape": list(pred_depth.shape),
        "pred_depth_valid_ratio": float(valid.sum() / pred_depth.size),
        "depth_mae_m": float(np.mean(diff_abs)) if valid.any() else None,
        "depth_rmse_m": float(np.sqrt(np.mean((pred_depth[valid] - gt_depth[valid]) ** 2))) if valid.any() else None,
        "depth_error_median_m": float(np.median(diff_abs)) if valid.any() else None,
        "depth_error_p90_m": float(np.percentile(diff_abs, 90)) if valid.any() else None,
        "depth_error_p95_m": float(np.percentile(diff_abs, 95)) if valid.any() else None,
        "mask_source": metadata.get("mask_source"),
        "mask_point_count": int(mask.sum()),
        "error": "",
        "traceback": "",
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
