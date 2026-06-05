#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_spatial_calib.io_utils import OUTPUT_ROOT, sample_dirs, write_json  # noqa: E402


def rel(path):
    try:
        return Path(path).resolve().relative_to((OUTPUT_ROOT / "reports").resolve())
    except Exception:
        return Path(path)


def main():
    reports = OUTPUT_ROOT / "reports"
    dataset = OUTPUT_ROOT / "dataset"
    pred = OUTPUT_ROOT / "predictions"
    viz = OUTPUT_ROOT / "visualizations"
    rows_path = reports / "spatial_localization_report.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    samples = sample_dirs(dataset)
    sample = samples[0] if samples else None
    sample_id = sample.name if sample else (rows[0]["sample_id"] if rows else "")
    row = rows[0] if rows else {}
    runtime_path = pred / sample_id / "foundation_stereo_runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}

    lines = [
        "# Stereo Spatial Calibration Visual Debug Report",
        "",
        f"- sample_id: `{sample_id}`",
        f"- backend: `{row.get('backend', '')}`",
        f"- native_capture_method: `{row.get('native_capture_method', '')}`",
        f"- mask_source: `{row.get('mask_source', '')}`",
        f"- segmentation_available: `{row.get('segmentation_available', '')}`",
        f"- depth_source: `{row.get('depth_source', '')}`",
        f"- FoundationStereo checkpoint_available: `{runtime.get('checkpoint_available', False)}`",
        f"- FoundationStereo inference_success: `{runtime.get('inference_success', False)}`",
        "",
        "## Coordinates",
        "",
        f"- center_camera_m: `{row.get('center_pred_camera_m', row.get('center_camera_m', ''))}`",
        f"- center_world_m: `{row.get('center_pred_world_m', '')}`",
        f"- center_base_m: `{row.get('center_pred_base_m', '')}`",
        f"- center_gt_world_m: `{row.get('center_gt_world_m', '')}`",
        f"- error_l2_mm: `{row.get('error_l2_mm', '')}`",
        "",
        "## Images",
        "",
    ]
    image_paths = []
    if sample:
        image_paths += [sample / "left_rgb.png", sample / "right_rgb.png"]
    image_paths += [
        viz / f"{sample_id}_left_rgb_with_mask.png",
        viz / f"{sample_id}_gt_depth_color.png",
        pred / sample_id / "disparity_color.png",
        pred / sample_id / "pred_depth_color.png",
        viz / f"{sample_id}_depth_error_color.png",
        viz / f"{sample_id}_object_center_overlay.png",
    ]
    for path in image_paths:
        if path.exists():
            lines.append(f"![{path.name}]({path})")
            lines.append("")
    lines += [
        "## Point Clouds",
        "",
        f"- pointcloud_dir: `{OUTPUT_ROOT / 'pointclouds'}`",
        "",
    ]
    md_path = reports / "visual_debug_report.md"
    html_path = reports / "visual_debug_report.html"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    body = ["<html><body>", "<h1>Stereo Spatial Calibration Visual Debug Report</h1>"]
    for line in lines:
        if line.startswith("!["):
            name = line.split("](")[0][2:]
            path = line.split("](")[1].rstrip(")")
            body.append(f"<h3>{html.escape(name)}</h3><img src='{html.escape(path)}' style='max-width:900px'>")
        elif line.startswith("- "):
            body.append(f"<p>{html.escape(line)}</p>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
    body.append("</body></html>")
    html_path.write_text("\n".join(body), encoding="utf-8")
    write_json(reports / "visual_debug_report_summary.json", {"markdown": str(md_path), "html": str(html_path)})
    print(f"Wrote {md_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
