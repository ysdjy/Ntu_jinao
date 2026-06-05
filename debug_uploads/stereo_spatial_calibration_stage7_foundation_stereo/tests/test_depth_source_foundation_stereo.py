from __future__ import annotations

from pathlib import Path

from stereo_spatial_calib.foundation_stereo_adapter import FoundationStereoAdapter
from stereo_spatial_calib.io_utils import read_json
from stereo_spatial_calib.isaac_data_collector import generate_synthetic_stereo_dataset


def test_foundation_stereo_depth_source_checkpoint_missing_is_not_gt(tmp_path: Path):
    dataset = tmp_path / "dataset"
    predictions = tmp_path / "predictions"
    sample = generate_synthetic_stereo_dataset(dataset, objects=["cube"], num_samples_per_object=1, width=64, height=48)[0]
    adapter = FoundationStereoAdapter(repo_path=tmp_path / "FoundationStereo")

    row = adapter.process_sample(sample, predictions, depth_source="foundation_stereo")

    pred_dir = predictions / "cube_000001"
    runtime = read_json(pred_dir / "foundation_stereo_runtime.json")
    assert row["skipped"] is True
    assert row["actual_depth_source"] == "foundation_stereo_checkpoint_missing"
    assert runtime["status"] == "checkpoint_missing"
    assert runtime["inference_success"] is False
    assert not (pred_dir / "pred_depth.npy").exists()
    assert not (pred_dir / "disparity.npy").exists()
