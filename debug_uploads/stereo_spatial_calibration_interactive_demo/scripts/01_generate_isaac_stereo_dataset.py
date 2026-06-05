#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_spatial_calib.io_utils import OUTPUT_ROOT  # noqa: E402
from stereo_spatial_calib.isaac_data_collector import (  # noqa: E402
    generate_isaac_native_stereo_dataset,
    generate_synthetic_stereo_dataset,
)
from stereo_spatial_calib.isaac_scene_builder import isaac_available  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(OUTPUT_ROOT / "dataset"))
    parser.add_argument("--objects", nargs="+", default=["cube", "sphere", "cylinder"])
    parser.add_argument("--num_samples_per_object", type=int, default=10)
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--baseline", type=float, default=0.10)
    parser.add_argument("--backend", choices=["synthetic", "isaac_native"], default="synthetic")
    parser.add_argument(
        "--native_capture_method",
        choices=["auto", "camera_class", "replicator_annotator", "tiled_camera"],
        default="auto",
    )
    parser.add_argument(
        "--mask_source",
        choices=[
            "auto",
            "native_instance",
            "native_instance_id",
            "native_semantic",
            "native_bbox_2d_tight",
            "projected_bbox_fallback",
        ],
        default="auto",
    )
    args = parser.parse_args()

    if args.backend == "synthetic":
        created = generate_synthetic_stereo_dataset(
            args.output_dir,
            objects=args.objects,
            num_samples_per_object=args.num_samples_per_object,
            width=args.resolution[0],
            height=args.resolution[1],
            baseline_m=args.baseline,
        )
    else:
        if os.environ.get("CONDA_DEFAULT_ENV") != "env_isaaclab":
            raise SystemExit(
                "isaac_native backend must be run inside env_isaaclab. "
                "Run: source /home1/banghai/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab"
            )
        if not isaac_available():
            err = "Isaac modules are not importable in the active Python runtime."
            log = OUTPUT_ROOT / "reports" / "isaac_native_backend_error.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(err + "\n", encoding="utf-8")
            raise SystemExit(err)
        created = generate_isaac_native_stereo_dataset(
            args.output_dir,
            reports_dir=OUTPUT_ROOT / "reports",
            objects=args.objects,
            num_samples_per_object=args.num_samples_per_object,
            width=args.resolution[0],
            height=args.resolution[1],
            baseline_m=args.baseline,
            native_capture_method=args.native_capture_method,
            mask_source=args.mask_source,
        )
    print(f"Generated {len(created)} samples under {args.output_dir}")
    if args.backend == "isaac_native":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
