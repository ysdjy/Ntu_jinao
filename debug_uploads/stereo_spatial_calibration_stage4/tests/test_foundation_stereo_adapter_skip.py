from __future__ import annotations

from pathlib import Path

from stereo_spatial_calib.foundation_stereo_adapter import FoundationStereoAdapter
from stereo_spatial_calib.io_utils import read_json
from stereo_spatial_calib.isaac_data_collector import generate_synthetic_stereo_dataset


def test_foundation_stereo_missing_checkpoint_falls_back_without_failure(tmp_path: Path):
    dataset = tmp_path / "dataset"
    predictions = tmp_path / "predictions"
    sample = generate_synthetic_stereo_dataset(dataset, objects=["cube"], num_samples_per_object=1, width=64, height=48)[0]
    adapter = FoundationStereoAdapter(repo_path=tmp_path / "missing_repo", checkpoint_path=tmp_path / "missing.pth")

    row = adapter.process_sample(sample, predictions, depth_source="foundation_stereo", allow_fallback=True)

    assert row["actual_depth_source"] == "gt_depth_fallback_for_foundation_stereo"
    runtime = read_json(predictions / "cube_000001" / "foundation_stereo_runtime.json")
    assert runtime["status"] == "skipped_fallback_gt"
    assert (predictions / "cube_000001" / "pred_depth.npy").exists()
    assert (predictions / "cube_000001" / "disparity.npy").exists()

