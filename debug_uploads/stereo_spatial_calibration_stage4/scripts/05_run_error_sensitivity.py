#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_spatial_calib.error_sensitivity import run_error_sensitivity  # noqa: E402
from stereo_spatial_calib.io_utils import OUTPUT_ROOT  # noqa: E402
from stereo_spatial_calib.isaac_data_collector import generate_synthetic_stereo_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["synthetic", "isaac_native"], default="synthetic")
    parser.add_argument("--objects", nargs="+", default=["cube", "sphere", "cylinder"])
    parser.add_argument("--num_samples_per_object", type=int, default=1)
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480])
    parser.add_argument("--baseline", type=float, default=0.10)
    parser.add_argument("--dataset_dir", default=str(OUTPUT_ROOT / "dataset"))
    parser.add_argument("--reports_dir", default=str(OUTPUT_ROOT / "reports"))
    parser.add_argument("--visualizations_dir", default=str(OUTPUT_ROOT / "visualizations"))
    parser.add_argument("--depth_noise_std_list", nargs="+", type=float, default=[0.0, 0.002, 0.005, 0.010])
    parser.add_argument("--depth_bias_list", nargs="+", type=float, default=[-0.010, 0.0, 0.010])
    parser.add_argument("--mask_shift_px_list", nargs="+", type=int, default=[-5, 0, 5])
    parser.add_argument("--mask_morphology_list", nargs="+", default=["none", "erode", "dilate"])
    parser.add_argument("--mask_kernel_size_list", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--cam_trans_noise_std_list", nargs="+", type=float, default=[0.0, 0.002, 0.005])
    parser.add_argument("--cam_rot_noise_deg_list", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--fx_relative_error_list", nargs="+", type=float, default=[0.0, 0.005, 0.010])
    parser.add_argument("--fy_relative_error_list", nargs="+", type=float, default=None)
    parser.add_argument("--cx_shift_px_list", nargs="+", type=float, default=[0.0, 2.0])
    parser.add_argument("--cy_shift_px_list", nargs="+", type=float, default=[0.0, 2.0])
    parser.add_argument("--baseline_relative_error_list", nargs="+", type=float, default=[0.0, 0.005, 0.010])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.backend != "synthetic":
        raise SystemExit("error sensitivity currently uses saved dataset arrays; run isaac_native capture first, then pass its dataset_dir.")
    generate_synthetic_stereo_dataset(
        args.dataset_dir,
        objects=args.objects,
        num_samples_per_object=args.num_samples_per_object,
        width=args.resolution[0],
        height=args.resolution[1],
        baseline_m=args.baseline,
        seed=args.seed,
    )
    rows = run_error_sensitivity(
        args.dataset_dir,
        args.reports_dir,
        args.visualizations_dir,
        depth_noise_std_list=args.depth_noise_std_list,
        depth_bias_list=args.depth_bias_list,
        mask_shift_px_list=args.mask_shift_px_list,
        mask_morphology_list=args.mask_morphology_list,
        mask_kernel_size_list=args.mask_kernel_size_list,
        cam_trans_noise_std_list=args.cam_trans_noise_std_list,
        cam_rot_noise_deg_list=args.cam_rot_noise_deg_list,
        fx_relative_error_list=args.fx_relative_error_list,
        fy_relative_error_list=args.fy_relative_error_list,
        cx_shift_px_list=args.cx_shift_px_list,
        cy_shift_px_list=args.cy_shift_px_list,
        baseline_relative_error_list=args.baseline_relative_error_list,
        seed=args.seed,
    )
    print(f"Wrote {len(rows)} sensitivity rows under {args.reports_dir}")


if __name__ == "__main__":
    main()
