#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def run_step(script_name, args):
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)] + args
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples_per_object", type=int, default=3)
    parser.add_argument("--objects", nargs="+", default=["cube", "sphere", "cylinder"])
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--baseline", type=float, default=0.10)
    parser.add_argument("--use_gt_depth_as_prediction", type=str2bool, default=False)
    parser.add_argument("--backend", choices=["synthetic", "isaac_native"], default="synthetic")
    parser.add_argument(
        "--native_capture_method",
        choices=["auto", "camera_class", "replicator_annotator", "tiled_camera"],
        default="auto",
    )
    parser.add_argument("--depth_source", choices=["gt", "foundation_stereo", "noisy_gt"], default=None)
    parser.add_argument("--depth_noise_std_m", type=float, default=0.005)
    parser.add_argument("--depth_noise_bias_m", type=float, default=0.0)
    parser.add_argument("--foundation_stereo_repo", default=None)
    parser.add_argument("--foundation_stereo_model_dir", default=None)
    parser.add_argument("--foundation_stereo_ckpt", default=None)
    parser.add_argument("--foundation_stereo_cfg", default=None)
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
    parser.add_argument("--clean_outputs", type=str2bool, default=True)
    args = parser.parse_args()
    depth_source = args.depth_source
    if depth_source is None:
        depth_source = "gt" if args.use_gt_depth_as_prediction else "foundation_stereo"

    dataset_dir = PROJECT_ROOT / "outputs" / "dataset"
    pred_dir = PROJECT_ROOT / "outputs" / "predictions"
    pc_dir = PROJECT_ROOT / "outputs" / "pointclouds"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    viz_dir = PROJECT_ROOT / "outputs" / "visualizations"

    if args.clean_outputs:
        for path in [dataset_dir, pred_dir, pc_dir, reports_dir, viz_dir]:
            if path.exists():
                shutil.rmtree(path)

    run_step("00_check_environment.py", [])
    run_step(
        "01_generate_isaac_stereo_dataset.py",
        [
            "--output_dir",
            str(dataset_dir),
            "--objects",
            *args.objects,
            "--num_samples_per_object",
            str(args.num_samples_per_object),
            "--resolution",
            str(args.resolution[0]),
            str(args.resolution[1]),
            "--baseline",
            str(args.baseline),
            "--backend",
            args.backend,
            "--native_capture_method",
            args.native_capture_method,
            "--mask_source",
            args.mask_source,
        ],
    )
    run_step(
        "02_run_foundation_stereo.py",
        [
            "--dataset_dir",
            str(dataset_dir),
            "--predictions_dir",
            str(pred_dir),
            "--use_gt_depth_as_prediction",
            str(args.use_gt_depth_as_prediction).lower(),
            "--depth_source",
            depth_source,
            "--depth_noise_std_m",
            str(args.depth_noise_std_m),
            "--depth_noise_bias_m",
            str(args.depth_noise_bias_m),
            "--foundation_stereo_repo",
            str(args.foundation_stereo_repo or ""),
            "--foundation_stereo_model_dir",
            str(args.foundation_stereo_model_dir or ""),
            "--foundation_stereo_ckpt",
            str(args.foundation_stereo_ckpt or ""),
            "--foundation_stereo_cfg",
            str(args.foundation_stereo_cfg or ""),
        ],
    )
    summary_path = pred_dir / "prediction_summary.json"
    if depth_source == "foundation_stereo" and summary_path.exists():
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
        if rows and all(row.get("skipped") for row in rows):
            print("\nFoundationStereo pipeline skipped before pointcloud reconstruction.")
            print("Reason:", ", ".join(sorted({str(row.get("skip_reason")) for row in rows})))
            print(f"Prediction summary: {summary_path}")
            print(f"Reports: {reports_dir}")
            return 0
    run_step(
        "03_reconstruct_pointcloud.py",
        ["--dataset_dir", str(dataset_dir), "--predictions_dir", str(pred_dir), "--pointcloud_dir", str(pc_dir)],
    )
    run_step(
        "04_evaluate_spatial_localization.py",
        [
            "--dataset_dir",
            str(dataset_dir),
            "--predictions_dir",
            str(pred_dir),
            "--reports_dir",
            str(reports_dir),
            "--visualizations_dir",
            str(viz_dir),
        ],
    )
    print("\nPipeline complete.")
    print(f"Reports: {reports_dir}")
    print(f"Pointclouds: {pc_dir}")
    print(f"Visualizations: {viz_dir}")


if __name__ == "__main__":
    main()
