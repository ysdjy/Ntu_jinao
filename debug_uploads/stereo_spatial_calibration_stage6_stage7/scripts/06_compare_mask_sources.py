#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_spatial_calib.evaluator import evaluate_sample  # noqa: E402
from stereo_spatial_calib.foundation_stereo_adapter import FoundationStereoAdapter  # noqa: E402
from stereo_spatial_calib.io_utils import OUTPUT_ROOT, load_image_rgb, save_image, write_csv, write_json  # noqa: E402
from stereo_spatial_calib.segmentation_utils import load_object_mask, mask_to_bbox  # noqa: E402
from stereo_spatial_calib.visualization import overlay_mask  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", nargs="+", default=["cube"])
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--baseline", type=float, default=0.10)
    args = parser.parse_args()

    sources = [
        "projected_bbox_fallback",
        "native_bbox_2d_tight",
        "native_semantic",
        "native_instance",
        "native_instance_id",
    ]
    root = OUTPUT_ROOT / "mask_source_comparison"
    reports = OUTPUT_ROOT / "reports"
    viz = OUTPUT_ROOT / "visualizations"
    rows = []
    overlays = []
    adapter = FoundationStereoAdapter()
    for source in sources:
        dataset_dir = root / f"dataset_{source}"
        pred_dir = root / f"predictions_{source}"
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "01_generate_isaac_stereo_dataset.py"),
                "--output_dir",
                str(dataset_dir),
                "--objects",
                *args.objects,
                "--num_samples_per_object",
                "1",
                "--resolution",
                str(args.resolution[0]),
                str(args.resolution[1]),
                "--baseline",
                str(args.baseline),
                "--backend",
                "isaac_native",
                "--native_capture_method",
                "camera_class",
                "--mask_source",
                source,
            ],
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        samples = sorted([p for p in dataset_dir.iterdir() if (p / "metadata.json").exists()])
        sample = samples[0]
        adapter.process_sample(sample, pred_dir, depth_source="gt")
        row = evaluate_sample(sample, pred_dir)
        mask = load_object_mask(sample)
        rgb = load_image_rgb(sample / "left_rgb.png")
        overlay = overlay_mask(rgb, mask)
        overlays.append(overlay)
        save_image(viz / f"mask_source_{source}_overlay.png", overlay)
        row.update(
            {
                "requested_mask_source": source,
                "actual_mask_source": row.get("mask_source"),
                "mask_pixel_count": int(mask.sum()),
                "mask_bbox": mask_to_bbox(mask),
            }
        )
        rows.append(row)

    if overlays:
        h, w = overlays[0].shape[:2]
        grid = np.zeros((h, w * len(overlays), 3), dtype=np.uint8)
        for idx, img in enumerate(overlays):
            grid[:, idx * w : (idx + 1) * w] = img
        save_image(viz / "mask_source_comparison_grid.png", grid)

    fieldnames = [
        "requested_mask_source",
        "actual_mask_source",
        "sample_id",
        "object_type",
        "mask_pixel_count",
        "mask_bbox",
        "center_pred_world_m",
        "center_gt_world_m",
        "error_l2_mm",
        "depth_rmse_m",
        "valid_depth_ratio",
    ]
    write_csv(reports / "mask_source_comparison_report.csv", rows, fieldnames=fieldnames)
    write_json(reports / "mask_source_comparison_report.json", rows)
    print(f"Wrote {reports / 'mask_source_comparison_report.csv'}")
    for row in rows:
        print(f"{row['requested_mask_source']} -> {row.get('actual_mask_source')}: {row['error_l2_mm']:.3f} mm")


if __name__ == "__main__":
    main()
