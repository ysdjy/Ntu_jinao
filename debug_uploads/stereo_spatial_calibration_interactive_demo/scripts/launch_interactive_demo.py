#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_spatial_calib.interactive_demo_runner import run_interactive_capture_and_infer  # noqa: E402
from stereo_spatial_calib.io_utils import write_json  # noqa: E402


def _launch_gui(args):
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=False, enable_cameras=True)
    app = app_launcher.app
    import omni.ui as ui
    status = {"text": "Idle", "last_result": {}}

    def run_once():
        status["text"] = "Running FoundationStereo..."
        try:
            result = run_interactive_capture_and_infer(
                target_object=args.target_object,
                baseline_m=args.baseline,
                resolution=tuple(args.resolution),
                depth_source="foundation_stereo",
                mask_source=args.mask_source,
                use_existing_scene=True,
            )
            status["last_result"] = result
            status["text"] = f"Done: error_l2_mm={result.get('error_l2_mm')}"
        except Exception as exc:
            status["text"] = f"Failed: {type(exc).__name__}: {exc}"

    with ui.Window("Stereo Spatial Calibration Demo", width=420, height=560):
        with ui.VStack(spacing=8):
            ui.Label("Scene Controls")
            ui.Button("Create / Reset Demo Scene", clicked_fn=lambda: None)
            ui.Button("Create Stereo Rig", clicked_fn=lambda: None)
            ui.Button("Repair Stereo Rig", clicked_fn=lambda: None)
            ui.Button("Randomize Target Object Pose", clicked_fn=lambda: None)
            ui.Label(f"Target Object: {args.target_object}")
            ui.Spacer(height=8)
            ui.Label("Camera Controls")
            ui.Label(f"baseline_m: {args.baseline}")
            ui.Label(f"resolution: {args.resolution[0]} x {args.resolution[1]}")
            ui.Button("Validate Stereo Rig", clicked_fn=lambda: None)
            ui.Spacer(height=8)
            ui.Label("Inference Controls")
            ui.Button("Capture Stereo RGB", clicked_fn=lambda: None)
            ui.Button("Run FoundationStereo", clicked_fn=run_once)
            ui.Button("Run Full Localization", clicked_fn=run_once)
            ui.Button("Generate Visual Report", clicked_fn=lambda: None)
            ui.Spacer(height=8)
            ui.Label("Results Panel")
            ui.Label("Status updates are printed to the terminal and written to outputs/interactive_demo/latest/report.json")

    print("Stereo Spatial Calibration Demo GUI launched.")
    print("Use the Isaac Sim window panel: Stereo Spatial Calibration Demo")
    while app.is_running():
        app.update()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--headless_smoke_test", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--baseline", type=float, default=0.10)
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480])
    parser.add_argument("--target_object", default="cube", choices=["cube", "sphere", "cylinder"])
    parser.add_argument(
        "--mask_source",
        default="native_instance",
        choices=["auto", "native_instance", "native_instance_id", "native_semantic", "native_bbox_2d_tight", "projected_bbox_fallback"],
    )
    args = parser.parse_args()
    os.environ.setdefault("ENABLE_CAMERAS", "1")

    if args.headless_smoke_test:
        out = PROJECT_ROOT / "outputs" / "interactive_demo" / "latest"
        result = run_interactive_capture_and_infer(
            target_object=args.target_object,
            baseline_m=args.baseline,
            resolution=tuple(args.resolution),
            output_dir=out,
            depth_source="foundation_stereo",
            mask_source=args.mask_source,
            use_existing_scene=False,
        )
        write_json(PROJECT_ROOT / "outputs" / "reports" / "interactive_demo_smoke_report.json", result)
        print(json.dumps(result, indent=2))
        return 0

    if args.gui:
        _launch_gui(args)
        return 0

    parser.error("Use --headless_smoke_test or --gui")


if __name__ == "__main__":
    raise SystemExit(main())
