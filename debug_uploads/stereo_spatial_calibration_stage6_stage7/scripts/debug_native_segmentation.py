#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_spatial_calib.io_utils import OUTPUT_ROOT, colorize_scalar, read_json, save_image, write_json  # noqa: E402
from stereo_spatial_calib.isaac_data_collector import generate_isaac_native_stereo_dataset  # noqa: E402
from stereo_spatial_calib.segmentation_utils import mask_to_bbox  # noqa: E402
from stereo_spatial_calib.visualization import overlay_mask  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--baseline", type=float, default=0.10)
    parser.add_argument("--mask_source", default="auto")
    args = parser.parse_args()

    out = OUTPUT_ROOT / "debug_native_segmentation"
    reports = OUTPUT_ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "debug_native_segmentation_report.json"
    final_report_path = reports / "debug_native_segmentation_report.json"
    report = {
        "status": "failed",
        "python": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "enable_cameras": os.environ.get("ENABLE_CAMERAS"),
        "frame_keys": [],
        "available_segmentation_keys": [],
        "semantic_label_set_success": False,
        "instance_ids_unique_count": 0,
        "target_instance_id": None,
        "target_mask_pixel_count": 0,
        "mask_source": None,
        "error": "",
        "traceback": "",
    }
    try:
        sample_root = out / "dataset"
        samples = generate_isaac_native_stereo_dataset(
            sample_root,
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
        rgb = np.asarray(Image.open(sample / "left_rgb.png").convert("RGB"))
        depth = np.load(sample / "gt_depth_left.npy")
        mask = np.load(sample / "gt_instance_mask.npy").astype(bool)
        metadata = read_json(sample / "metadata.json")
        frame_summary = metadata.get("native_frame_summary", {})
        frame_keys = sorted(frame_summary.keys())
        seg_keys = [k for k in frame_keys if "segmentation" in k or "bounding_box" in k]
        valid = np.isfinite(depth) & (depth > 0.0)

        save_image(out / "rgb.png", rgb)
        np.save(out / "depth.npy", depth.astype(np.float32))
        save_image(out / "depth_color.png", colorize_scalar(depth))
        write_json(out / "frame_keys.json", frame_summary)
        np.save(out / "semantic_or_instance_raw.npy", mask.astype(np.int32))
        save_image(out / "semantic_or_instance_color.png", colorize_scalar(mask.astype(np.float32)))
        np.save(out / "target_mask.npy", mask.astype(np.uint8))
        save_image(out / "target_mask.png", mask.astype(np.uint8) * 255)
        save_image(out / "rgb_with_mask_overlay.png", overlay_mask(rgb, mask))
        bbox = mask_to_bbox(mask)
        bbox_img = rgb.copy()
        if bbox:
            x0, y0, x1, y1 = bbox
            bbox_img[y0 : y1 + 1, [x0, x1]] = [0, 255, 0]
            bbox_img[[y0, y1], x0 : x1 + 1] = [0, 255, 0]
        save_image(out / "bbox_debug.png", bbox_img)

        report.update(
            {
                "status": "success",
                "frame_keys": frame_keys,
                "rgb_mean": float(np.mean(rgb)),
                "depth_valid_ratio": float(valid.sum() / valid.size),
                "available_segmentation_keys": seg_keys,
                "semantic_label_set_success": bool(metadata.get("semantic_label_set_success", False)),
                "instance_ids_unique_count": int(np.unique(mask.astype(np.uint8)).size),
                "target_instance_id": metadata.get("target_instance_id"),
                "target_mask_pixel_count": int(mask.sum()),
                "mask_source": metadata.get("mask_source"),
                "sample_dir": str(sample),
            }
        )
    except Exception as exc:
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
    write_json(report_path, report)
    write_json(final_report_path, report)
    print(json.dumps(report, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if report["status"] == "success" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
