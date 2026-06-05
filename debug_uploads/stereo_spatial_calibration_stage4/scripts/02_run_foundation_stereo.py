#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_spatial_calib.foundation_stereo_adapter import FoundationStereoAdapter  # noqa: E402
from stereo_spatial_calib.io_utils import OUTPUT_ROOT, sample_dirs, write_json  # noqa: E402


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=str(OUTPUT_ROOT / "dataset"))
    parser.add_argument("--predictions_dir", default=str(OUTPUT_ROOT / "predictions"))
    parser.add_argument("--repo_path", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--use_gt_depth_as_prediction", type=str2bool, default=False)
    parser.add_argument("--depth_source", choices=["gt", "foundation_stereo", "noisy_gt"], default=None)
    parser.add_argument("--depth_noise_std_m", type=float, default=0.005)
    parser.add_argument("--depth_noise_bias_m", type=float, default=0.0)
    parser.add_argument("--allow_fallback", type=str2bool, default=True)
    args = parser.parse_args()

    adapter = FoundationStereoAdapter(args.repo_path, args.checkpoint_path)
    print(f"FoundationStereo status: {adapter.status()}")
    rows = []
    for sample_dir in sample_dirs(args.dataset_dir):
        row = adapter.process_sample(
            sample_dir,
            args.predictions_dir,
            use_gt_depth_as_prediction=args.use_gt_depth_as_prediction,
            depth_source=args.depth_source,
            depth_noise_std_m=args.depth_noise_std_m,
            depth_noise_bias_m=args.depth_noise_bias_m,
            allow_fallback=args.allow_fallback,
        )
        rows.append(row)
        print(f"Processed {sample_dir.name}: {row['actual_depth_source']}")
    write_json(Path(args.predictions_dir) / "prediction_summary.json", rows)
    print(f"Wrote predictions for {len(rows)} samples under {args.predictions_dir}")


if __name__ == "__main__":
    main()
