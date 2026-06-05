from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_download_script_dry_run_does_not_fail(tmp_path):
    repo = tmp_path / "FoundationStereo"
    repo.mkdir()
    (repo / "readme.md").write_text(
        "[23-51-11](https://drive.google.com/drive/folders/example)\n",
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "download_foundation_stereo_checkpoint.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--foundation_stereo_repo",
            str(repo),
            "--foundation_stereo_model_dir",
            str(repo / "pretrained_models" / "23-51-11"),
            "--dry_run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "dry_run" in result.stdout
