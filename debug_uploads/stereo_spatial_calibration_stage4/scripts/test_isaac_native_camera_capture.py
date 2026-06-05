#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import traceback
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_spatial_calib.io_utils import OUTPUT_ROOT, read_json, write_json  # noqa: E402
from stereo_spatial_calib.isaac_data_collector import generate_isaac_native_stereo_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native_capture_method",
        choices=["auto", "camera_class", "replicator_annotator", "tiled_camera"],
        default="auto",
    )
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--baseline", type=float, default=0.10)
    args = parser.parse_args()

    reports_dir = OUTPUT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "isaac_native_camera_capture_report.json"
    error_path = reports_dir / "isaac_native_backend_error.log"
    if os.environ.get("CONDA_DEFAULT_ENV") != "env_isaaclab":
        msg = (
            "test_isaac_native_camera_capture.py must run inside env_isaaclab. "
            "Run: source /home1/banghai/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab"
        )
        error_path.write_text(msg + "\n", encoding="utf-8")
        print(msg)
        return 2
    try:
        samples = generate_isaac_native_stereo_dataset(
            OUTPUT_ROOT / "dataset" / "isaac_native_camera_test",
            reports_dir=reports_dir,
            objects=["cube"],
            num_samples_per_object=1,
            width=args.resolution[0],
            height=args.resolution[1],
            baseline_m=args.baseline,
            seed=42,
            native_capture_method=args.native_capture_method,
        )
        metadata = read_json(samples[0] / "metadata.json")
        report = {
            "status": "success",
            "sample_dir": str(samples[0]),
            "left_rgb": str(samples[0] / "left_rgb.png"),
            "right_rgb": str(samples[0] / "right_rgb.png"),
            "gt_depth_left": str(samples[0] / "gt_depth_left.npy"),
            "left_camera_intrinsics": metadata["left_camera_intrinsics"],
            "baseline_m": metadata["baseline_m"],
            "T_world_left_cam": metadata["T_world_left_cam"],
            "mask_source": metadata.get("mask_source"),
            "native_capture_method": metadata.get("native_capture_method"),
            "segmentation_available": metadata.get("segmentation_available"),
            "gt_instance_mask": str(samples[0] / "gt_instance_mask.npy"),
        }
        write_json(report_path, report)
        print(json.dumps(report, indent=2))
        return 0
    except Exception:
        tb = traceback.format_exc()
        error_path.write_text(tb, encoding="utf-8")
        report = {"status": "failed", "error_log": str(error_path)}
        write_json(report_path, report)
        print(tb)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
