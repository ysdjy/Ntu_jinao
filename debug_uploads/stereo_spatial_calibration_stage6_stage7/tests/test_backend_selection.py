from __future__ import annotations

from pathlib import Path

from stereo_spatial_calib.io_utils import read_json
from stereo_spatial_calib.isaac_data_collector import generate_synthetic_stereo_dataset


def test_synthetic_backend_is_always_available(tmp_path: Path):
    samples = generate_synthetic_stereo_dataset(
        tmp_path / "dataset",
        objects=["cube", "sphere"],
        num_samples_per_object=1,
        width=64,
        height=48,
        baseline_m=0.10,
    )
    assert len(samples) == 2
    for sample in samples:
        metadata = read_json(Path(sample) / "metadata.json")
        assert metadata["backend"] == "synthetic_rectified_stereo"
        assert (Path(sample) / "left_rgb.png").exists()
        assert (Path(sample) / "right_rgb.png").exists()
        assert (Path(sample) / "gt_depth_left.npy").exists()
        assert (Path(sample) / "gt_instance_mask.npy").exists()

