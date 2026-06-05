#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_spatial_calib.foundation_stereo_adapter import FoundationStereoAdapter  # noqa: E402
from stereo_spatial_calib.io_utils import OUTPUT_ROOT, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", default=str(OUTPUT_ROOT / "dataset" / "cube_000001"))
    parser.add_argument("--predictions_dir", default=str(OUTPUT_ROOT / "predictions"))
    parser.add_argument("--repo_path", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    args = parser.parse_args()

    adapter = FoundationStereoAdapter(args.repo_path, args.checkpoint_path)
    status = adapter.status()
    if not adapter.ready:
        out = Path(args.predictions_dir) / "foundation_stereo_single_pair_skip.json"
        write_json(
            out,
            {
                "status": "skipped",
                "reason": "FoundationStereo runtime is not ready; checkpoint or torch dependency is missing.",
                "adapter_status": status,
            },
        )
        print("SKIP: FoundationStereo runtime is not ready.")
        print(json.dumps(status, indent=2))
        return 0

    row = adapter.process_sample(
        args.sample_dir,
        args.predictions_dir,
        depth_source="foundation_stereo",
        allow_fallback=False,
    )
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
