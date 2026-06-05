#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_spatial_calib.foundation_stereo_adapter import (  # noqa: E402
    clone_foundation_stereo,
    find_checkpoint,
    find_foundation_stereo_repo,
)
from stereo_spatial_calib.io_utils import write_json  # noqa: E402
from stereo_spatial_calib.runtime_diagnostics import (  # noqa: E402
    build_stage2_diagnosis,
    runtime_recommendation_markdown,
)


def check_import(module_name):
    try:
        mod = __import__(module_name)
        return {"available": True, "version": getattr(mod, "__version__", None)}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def main():
    repo = find_foundation_stereo_repo()
    clone_status = None
    if repo is None:
        repo, clone_status = clone_foundation_stereo()

    report = build_stage2_diagnosis(repo_path=repo)
    report["foundation_stereo"]["clone_status"] = clone_status
    report["foundation_stereo"]["fallback_available"] = True
    report["python"] = {
        "executable": report["current_python"]["executable"],
        "version": report["current_python"]["version"],
        "conda_default_env": report["current_python"]["conda_default_env"],
    }
    if not report["modules"]["open3d"]["available"]:
        report["open3d_install_hint"] = "pip install open3d; pipeline still writes ASCII PLY and NPY without open3d."
    if not report["foundation_stereo"]["ready"]:
        report["foundation_stereo"]["hint"] = (
            "FoundationStereo repo/checkpoint/runtime not fully ready. Use --depth_source gt "
            "or --depth_source noisy_gt for geometry-chain validation until a checkpoint is installed."
        )

    out = PROJECT_ROOT / "outputs" / "reports" / "environment_report.json"
    stage2_out = PROJECT_ROOT / "outputs" / "reports" / "stage2_environment_diagnosis.json"
    recommendation_out = PROJECT_ROOT / "outputs" / "reports" / "stage2_runtime_recommendation.md"
    write_json(out, report)
    write_json(stage2_out, report)
    recommendation_out.parent.mkdir(parents=True, exist_ok=True)
    recommendation_out.write_text(runtime_recommendation_markdown(report), encoding="utf-8")
    import json

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")
    print(f"Wrote {stage2_out}")
    print(f"Wrote {recommendation_out}")


if __name__ == "__main__":
    main()
