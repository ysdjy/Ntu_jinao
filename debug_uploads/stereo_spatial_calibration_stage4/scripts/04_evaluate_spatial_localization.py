#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_spatial_calib.evaluator import evaluate_sample  # noqa: E402
from stereo_spatial_calib.io_utils import OUTPUT_ROOT, sample_dirs, write_csv, write_json  # noqa: E402
from stereo_spatial_calib.visualization import write_sample_visualizations  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=str(OUTPUT_ROOT / "dataset"))
    parser.add_argument("--predictions_dir", default=str(OUTPUT_ROOT / "predictions"))
    parser.add_argument("--reports_dir", default=str(OUTPUT_ROOT / "reports"))
    parser.add_argument("--visualizations_dir", default=str(OUTPUT_ROOT / "visualizations"))
    parser.add_argument("--method", default="pointcloud_median", choices=["pointcloud_median", "pointcloud_mean_filtered"])
    args = parser.parse_args()

    rows = []
    for sample_dir in sample_dirs(args.dataset_dir):
        row = evaluate_sample(sample_dir, args.predictions_dir, method=args.method)
        rows.append(row)
        write_sample_visualizations(sample_dir, args.predictions_dir, row, args.visualizations_dir)
        print(f"{row['sample_id']}: error_l2_mm={row['error_l2_mm']:.3f}")

    report_dir = Path(args.reports_dir)
    csv_path = report_dir / "spatial_localization_report.csv"
    json_path = report_dir / "spatial_localization_report.json"
    stage2_csv_path = report_dir / "stage2_foundation_stereo_report.csv"
    stage2_json_path = report_dir / "stage2_foundation_stereo_report.json"
    stage4_csv_path = report_dir / "stage4_isaac_native_report.csv"
    stage4_json_path = report_dir / "stage4_isaac_native_report.json"
    fieldnames = [
        "sample_id",
        "object_type",
        "backend",
        "native_capture_method",
        "mask_source",
        "segmentation_available",
        "depth_source",
        "requested_depth_source",
        "method",
        "center_camera_m",
        "center_world_m",
        "center_base_m",
        "center_pred_camera_m",
        "center_pred_world_m",
        "center_pred_base_m",
        "gt_center_world_m",
        "gt_center_base_m",
        "center_gt_world_m",
        "error_x_m",
        "error_y_m",
        "error_z_m",
        "error_l2_m",
        "error_l2_mm",
        "depth_mae_m",
        "depth_rmse_m",
        "valid_depth_ratio",
        "mask_point_count",
        "center_pixel_u",
        "center_pixel_v",
        "foundation_stereo_runtime_ms",
    ]
    stage2_fieldnames = [
        "sample_id",
        "object_type",
        "backend",
        "depth_source",
        "center_pred_camera_m",
        "center_pred_world_m",
        "center_pred_base_m",
        "center_gt_world_m",
        "error_x_m",
        "error_y_m",
        "error_z_m",
        "error_l2_mm",
        "depth_mae_m",
        "depth_rmse_m",
        "valid_depth_ratio",
        "mask_point_count",
        "foundation_stereo_runtime_ms",
    ]
    stage4_fieldnames = [
        "sample_id",
        "object_type",
        "backend",
        "native_capture_method",
        "mask_source",
        "segmentation_available",
        "depth_source",
        "center_pred_world_m",
        "center_gt_world_m",
        "error_l2_mm",
        "depth_rmse_m",
        "valid_depth_ratio",
        "mask_point_count",
    ]
    write_csv(csv_path, rows, fieldnames=fieldnames)
    write_json(json_path, rows)
    write_csv(stage2_csv_path, rows, fieldnames=stage2_fieldnames)
    write_json(stage2_json_path, rows)
    write_csv(stage4_csv_path, rows, fieldnames=stage4_fieldnames)
    write_json(stage4_json_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {stage2_csv_path}")
    print(f"Wrote {stage2_json_path}")
    print(f"Wrote {stage4_csv_path}")
    print(f"Wrote {stage4_json_path}")


if __name__ == "__main__":
    main()
