#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_spatial_calib.io_utils import write_json  # noqa: E402
from stereo_spatial_calib.runtime_diagnostics import (  # noqa: E402
    current_python_report,
    foundation_stereo_report,
    probe_conda_env,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    args = parser.parse_args()
    report = {
        "current_python": current_python_report(include_sys_path=True),
        "foundation_stereo": foundation_stereo_report(args.repo_path, args.checkpoint_path),
        "env_isaaclab_probe": probe_conda_env("env_isaaclab"),
    }
    out = PROJECT_ROOT / "outputs" / "reports" / "foundation_stereo_runtime_check.json"
    write_json(out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
