from __future__ import annotations

from pathlib import Path

from stereo_spatial_calib.evaluator import evaluate_sample
from stereo_spatial_calib.foundation_stereo_adapter import FoundationStereoAdapter
from stereo_spatial_calib.isaac_data_collector import generate_synthetic_stereo_dataset


def test_noisy_gt_depth_rmse_increases_with_noise_std(tmp_path: Path):
    dataset = tmp_path / "dataset"
    preds_zero = tmp_path / "pred_zero"
    preds_noisy = tmp_path / "pred_noisy"
    sample = generate_synthetic_stereo_dataset(dataset, objects=["cube"], num_samples_per_object=1, width=64, height=48)[0]
    adapter = FoundationStereoAdapter(repo_path=tmp_path / "missing_repo", checkpoint_path=tmp_path / "missing.pth")

    adapter.process_sample(sample, preds_zero, depth_source="noisy_gt", depth_noise_std_m=0.0)
    adapter.process_sample(sample, preds_noisy, depth_source="noisy_gt", depth_noise_std_m=0.02)

    zero = evaluate_sample(sample, preds_zero)
    noisy = evaluate_sample(sample, preds_noisy)
    assert zero["depth_rmse_m"] == 0.0
    assert noisy["depth_rmse_m"] > zero["depth_rmse_m"]

