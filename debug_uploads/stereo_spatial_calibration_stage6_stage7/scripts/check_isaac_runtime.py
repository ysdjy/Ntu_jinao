#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_spatial_calib.io_utils import write_json  # noqa: E402
from stereo_spatial_calib.runtime_diagnostics import (  # noqa: E402
    current_python_report,
    import_status,
    probe_conda_env,
)


def main():
    report = {
        "current_python": current_python_report(include_sys_path=True),
        "imports": {
            "isaacsim": import_status("isaacsim"),
            "isaaclab": import_status("isaaclab"),
            "omni": import_status("omni"),
            "isaacsim.simulation_app": import_status("isaacsim.simulation_app"),
        },
        "env_isaaclab_probe": probe_conda_env("env_isaaclab"),
    }
    out = PROJECT_ROOT / "outputs" / "reports" / "isaac_runtime_check.json"
    write_json(out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

