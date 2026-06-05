from pathlib import Path

from stereo_spatial_calib.foundation_stereo_adapter import FoundationStereoAdapter, expected_checkpoint_files, find_checkpoint


def test_missing_checkpoint_clean_skip(tmp_path):
    repo = tmp_path / "FoundationStereo"
    repo.mkdir()
    adapter = FoundationStereoAdapter(repo_path=repo)
    status = adapter.status()
    assert not adapter.ready
    assert status["checkpoint_exists"] is False


def test_expected_checkpoint_detection(tmp_path):
    repo = tmp_path / "FoundationStereo"
    ckpt_dir = repo / "pretrained_models" / "23-51-11"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "model_best_bp2.pth"
    cfg = ckpt_dir / "cfg.yaml"
    ckpt.write_bytes(b"dummy")
    cfg.write_text("valid_iters: 1\n", encoding="utf-8")
    assert find_checkpoint(repo) == ckpt
    assert expected_checkpoint_files(repo) == (ckpt, cfg)
    adapter = FoundationStereoAdapter(repo_path=repo)
    assert adapter.checkpoint_path == ckpt
