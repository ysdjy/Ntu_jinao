import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from stereo_spatial_calib.io_utils import OUTPUT_ROOT, save_image


def test_visual_report_generation_with_dummy_outputs():
    sample = OUTPUT_ROOT / "dataset" / "dummy_000001"
    pred = OUTPUT_ROOT / "predictions" / "dummy_000001"
    reports = OUTPUT_ROOT / "reports"
    sample.mkdir(parents=True, exist_ok=True)
    pred.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    save_image(sample / "left_rgb.png", rgb)
    save_image(sample / "right_rgb.png", rgb)
    np.save(sample / "gt_depth_left.npy", np.ones((8, 8), dtype=np.float32))
    np.save(sample / "gt_instance_mask.npy", np.ones((8, 8), dtype=np.uint8))
    (sample / "metadata.json").write_text(json.dumps({"sample_id": "dummy_000001"}), encoding="utf-8")
    np.save(pred / "pred_depth.npy", np.ones((8, 8), dtype=np.float32))
    (reports / "spatial_localization_report.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": "dummy_000001",
                    "backend": "synthetic",
                    "mask_source": "dummy",
                    "center_pred_camera_m": [0, 0, 1],
                    "center_pred_world_m": [0, 0, 1],
                    "center_pred_base_m": [0, 0, 1],
                    "center_gt_world_m": [0, 0, 1],
                    "error_l2_mm": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "07_generate_visual_debug_report.py"
    subprocess.run([sys.executable, str(script)], check=True)
    assert (reports / "visual_debug_report.md").exists()
    assert (reports / "visual_debug_report.html").exists()
