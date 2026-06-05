from __future__ import annotations

from pathlib import Path

from stereo_spatial_calib.foundation_stereo_adapter import FoundationStereoAdapter
from stereo_spatial_calib.io_utils import read_json
from stereo_spatial_calib.isaac_data_collector import generate_synthetic_stereo_dataset


def test_foundation_stereo_missing_checkpoint_clean_skips_without_fake_depth(tmp_path: Path):
    dataset = tmp_path / "dataset"
    predictions = tmp_path / "predictions"
    sample = generate_synthetic_stereo_dataset(dataset, objects=["cube"], num_samples_per_object=1, width=64, height=48)[0]
    adapter = FoundationStereoAdapter(repo_path=tmp_path / "missing_repo", checkpoint_path=tmp_path / "missing.pth")

    row = adapter.process_sample(sample, predictions, depth_source="foundation_stereo", allow_fallback=True)

    assert row["actual_depth_source"] == "foundation_stereo_checkpoint_missing"
    assert row["skipped"] is True
    runtime = read_json(predictions / "cube_000001" / "foundation_stereo_runtime.json")
    assert runtime["status"] == "checkpoint_missing"
    assert runtime["inference_success"] is False
    assert not (predictions / "cube_000001" / "pred_depth.npy").exists()
    assert not (predictions / "cube_000001" / "disparity.npy").exists()


def test_explicit_gt_depth_still_writes_prediction(tmp_path: Path):
    dataset = tmp_path / "dataset"
    predictions = tmp_path / "predictions"
    sample = generate_synthetic_stereo_dataset(dataset, objects=["cube"], num_samples_per_object=1, width=64, height=48)[0]
    adapter = FoundationStereoAdapter(repo_path=tmp_path / "missing_repo", checkpoint_path=tmp_path / "missing.pth")

    row = adapter.process_sample(sample, predictions, depth_source="gt")

    assert row["actual_depth_source"] == "gt_depth"
    assert (predictions / "cube_000001" / "pred_depth.npy").exists()
    assert (predictions / "cube_000001" / "disparity.npy").exists()
