#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_spatial_calib.foundation_stereo_checkpoint import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    checkpoint_status,
    default_model_dir,
    disk_space_report,
    official_model_links,
    write_checkpoint_layout_report,
)
from stereo_spatial_calib.io_utils import OUTPUT_ROOT, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation_stereo_repo", default=None)
    parser.add_argument("--foundation_stereo_model_dir", default=None)
    parser.add_argument("--timeout_s", type=int, default=600)
    parser.add_argument("--min_free_gb", type=float, default=8.0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    reports = OUTPUT_ROOT / "reports"
    report_path = reports / "foundation_stereo_checkpoint_download_report.json"
    layout_path = reports / "foundation_stereo_checkpoint_layout.md"
    model_dir = Path(args.foundation_stereo_model_dir).expanduser() if args.foundation_stereo_model_dir else default_model_dir(args.foundation_stereo_repo)
    status_before = checkpoint_status(
        foundation_stereo_repo=args.foundation_stereo_repo,
        foundation_stereo_model_dir=model_dir,
    )
    links = official_model_links(args.foundation_stereo_repo)
    disk = disk_space_report(model_dir)
    report = {
        "status": "unknown",
        "source": None,
        "official_links": links,
        "disk_space": disk,
        "checkpoint_before": status_before,
        "checkpoint_after": None,
        "manual_steps": status_before["manual_steps"],
        "error": "",
        "stdout_tail": "",
        "stderr_tail": "",
    }
    write_checkpoint_layout_report(layout_path, repo_path=args.foundation_stereo_repo)

    if status_before["checkpoint_ready"]:
        report["status"] = "already_ready"
        report["checkpoint_after"] = status_before
        write_json(report_path, report)
        print(json.dumps(report, indent=2))
        return 0

    if disk["free_gb"] < float(args.min_free_gb):
        report["status"] = "failed"
        report["error"] = f"Insufficient free disk space: {disk['free_gb']} GB < {args.min_free_gb} GB."
        report["checkpoint_after"] = status_before
        write_json(report_path, report)
        print(json.dumps(report, indent=2))
        return 0

    url = links.get(DEFAULT_MODEL_NAME)
    if not url:
        report["status"] = "failed"
        report["error"] = "No official 23-51-11 checkpoint link found in local FoundationStereo readme.md."
        report["checkpoint_after"] = status_before
        write_json(report_path, report)
        print(json.dumps(report, indent=2))
        return 0

    if args.dry_run:
        report["status"] = "dry_run"
        report["source"] = url
        report["checkpoint_after"] = status_before
        write_json(report_path, report)
        print(json.dumps(report, indent=2))
        return 0

    gdown = shutil.which("gdown")
    if gdown is None:
        report["status"] = "failed"
        report["source"] = url
        report["error"] = (
            "Official checkpoint link is a Google Drive folder and gdown is not installed. "
            "Manual browser download is required."
        )
        report["checkpoint_after"] = status_before
        write_json(report_path, report)
        print(json.dumps(report, indent=2))
        return 0

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [gdown, "--folder", url, "-O", str(model_dir.parent)]
    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
            timeout=max(30, int(args.timeout_s)),
        )
        report["runtime_s"] = round(time.perf_counter() - start, 3)
        report["source"] = url
        report["command"] = cmd
        report["stdout_tail"] = result.stdout[-4000:]
        report["stderr_tail"] = result.stderr[-4000:]
        status_after = checkpoint_status(
            foundation_stereo_repo=args.foundation_stereo_repo,
            foundation_stereo_model_dir=model_dir,
        )
        report["checkpoint_after"] = status_after
        if result.returncode == 0 and status_after["checkpoint_ready"]:
            report["status"] = "success"
        else:
            report["status"] = "failed"
            report["error"] = (
                f"gdown returned {result.returncode} or expected files are still missing. "
                f"Missing: {status_after['missing_files']}"
            )
    except subprocess.TimeoutExpired as exc:
        report["status"] = "failed"
        report["source"] = url
        report["runtime_s"] = round(time.perf_counter() - start, 3)
        report["error"] = f"Download timed out after {args.timeout_s}s."
        report["stdout_tail"] = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
        report["stderr_tail"] = (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""
        report["checkpoint_after"] = checkpoint_status(
            foundation_stereo_repo=args.foundation_stereo_repo,
            foundation_stereo_model_dir=model_dir,
        )

    write_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
