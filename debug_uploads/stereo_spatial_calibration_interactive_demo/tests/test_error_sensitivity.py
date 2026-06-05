from __future__ import annotations

from pathlib import Path

import numpy as np

from stereo_spatial_calib.camera_geometry import apply_depth_noise
from stereo_spatial_calib.error_sensitivity import run_error_sensitivity
from stereo_spatial_calib.io_utils import read_json
from stereo_spatial_calib.isaac_data_collector import generate_synthetic_stereo_dataset


def test_depth_bias_moves_pointcloud_z_positive():
    depth = np.full((3, 3), 1.0, dtype=np.float32)
    biased = apply_depth_noise(depth, std_m=0.0, bias_m=0.01, seed=7)
    np.testing.assert_allclose(biased, 1.01, atol=1e-6)


def test_error_sensitivity_writes_reports(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "viz"
    samples = generate_synthetic_stereo_dataset(
        dataset_dir,
        objects=["cube"],
        num_samples_per_object=1,
        width=64,
        height=48,
        baseline_m=0.10,
        seed=3,
    )
    assert len(samples) == 1
    metadata = read_json(Path(samples[0]) / "metadata.json")
    assert metadata["object_type"] == "cube"

    rows = run_error_sensitivity(
        dataset_dir,
        reports_dir,
        viz_dir,
        depth_noise_std_list=[0.0],
        depth_bias_list=[0.0, 0.01],
        mask_shift_px_list=[0],
        mask_morphology_list=["none"],
        mask_kernel_size_list=[3],
        cam_trans_noise_std_list=[0.0],
        cam_rot_noise_deg_list=[0.0],
        fx_relative_error_list=[0.0],
        fy_relative_error_list=[0.0],
        cx_shift_px_list=[0.0],
        cy_shift_px_list=[0.0],
        baseline_relative_error_list=[0.0, 0.01],
        seed=11,
    )

    assert rows
    assert (reports_dir / "error_sensitivity_report.csv").exists()
    assert (reports_dir / "error_sensitivity_report.json").exists()
    bias_rows = [r for r in rows if r["perturbation_type"] == "depth_bias_m"]
    zero_bias = next(r for r in bias_rows if r["perturbation_value"] == "0.0")
    pos_bias = next(r for r in bias_rows if r["perturbation_value"] == "0.01")
    assert pos_bias["depth_rmse_m"] > zero_bias["depth_rmse_m"]

