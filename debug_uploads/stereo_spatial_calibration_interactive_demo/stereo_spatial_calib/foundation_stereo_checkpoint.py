"""FoundationStereo repository and checkpoint layout helpers."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


DEFAULT_REPO_CANDIDATES = [
    Path.home() / "FoundationStereo",
    Path.home() / "IsaacLab" / "FoundationStereo",
    Path.home() / "IsaacLab" / "third_party" / "FoundationStereo",
]
DEFAULT_MODEL_NAME = "23-51-11"
DEFAULT_CKPT_NAME = "model_best_bp2.pth"
DEFAULT_CFG_NAME = "cfg.yaml"


def find_foundation_stereo_repo(candidates=None):
    for path in candidates or DEFAULT_REPO_CANDIDATES:
        path = Path(path).expanduser()
        if path.exists() and path.is_dir():
            return path
    return None


def foundation_readme_path(repo_path=None):
    repo = Path(repo_path).expanduser() if repo_path else find_foundation_stereo_repo()
    if repo is None:
        return None
    for name in ("README.md", "readme.md", "README.rst", "readme.rst"):
        path = repo / name
        if path.exists():
            return path
    return None


def default_model_dir(repo_path=None):
    repo = Path(repo_path).expanduser() if repo_path else find_foundation_stereo_repo()
    if repo is None:
        repo = DEFAULT_REPO_CANDIDATES[-1]
    return repo / "pretrained_models" / DEFAULT_MODEL_NAME


def checkpoint_status(
    foundation_stereo_repo=None,
    foundation_stereo_model_dir=None,
    foundation_stereo_ckpt=None,
    foundation_stereo_cfg=None,
):
    repo = Path(foundation_stereo_repo).expanduser() if foundation_stereo_repo else find_foundation_stereo_repo()
    model_dir = Path(foundation_stereo_model_dir).expanduser() if foundation_stereo_model_dir else default_model_dir(repo)
    ckpt = Path(foundation_stereo_ckpt).expanduser() if foundation_stereo_ckpt else model_dir / DEFAULT_CKPT_NAME
    cfg = Path(foundation_stereo_cfg).expanduser() if foundation_stereo_cfg else model_dir / DEFAULT_CFG_NAME
    missing = []
    if not ckpt.exists():
        missing.append(str(ckpt))
    if not cfg.exists():
        missing.append(str(cfg))
    readme = foundation_readme_path(repo) if repo else None
    return {
        "repo_path": str(repo) if repo else None,
        "repo_exists": bool(repo and repo.exists()),
        "model_dir": str(model_dir),
        "ckpt_path": str(ckpt),
        "cfg_path": str(cfg),
        "ckpt_exists": bool(ckpt.exists()),
        "cfg_exists": bool(cfg.exists()),
        "checkpoint_ready": bool(ckpt.exists() and cfg.exists()),
        "missing_files": missing,
        "readme_path": str(readme) if readme else None,
        "expected": [str(model_dir / DEFAULT_CKPT_NAME), str(model_dir / DEFAULT_CFG_NAME)],
        "manual_steps": manual_download_steps(model_dir),
    }


def manual_download_steps(model_dir=None):
    model_dir = Path(model_dir).expanduser() if model_dir else default_model_dir()
    return [
        "Download the FoundationStereo 23-51-11 model folder from the official FoundationStereo README link.",
        f"Create: {model_dir}",
        f"Place {DEFAULT_CKPT_NAME} at: {model_dir / DEFAULT_CKPT_NAME}",
        f"Place {DEFAULT_CFG_NAME} at: {model_dir / DEFAULT_CFG_NAME}",
        "Do not commit checkpoint files to git.",
    ]


def official_model_links(repo_path=None):
    readme = foundation_readme_path(repo_path)
    if readme is None:
        return {}
    text = readme.read_text(encoding="utf-8", errors="ignore")
    links = {}
    for model_name in ("23-51-11", "11-33-40"):
        match = re.search(rf"\[{re.escape(model_name)}\]\(([^)]+)\)", text)
        if match:
            links[model_name] = match.group(1)
    return links


def disk_space_report(path):
    target = Path(path).expanduser()
    probe = target if target.exists() else target.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {
        "path": str(probe),
        "free_bytes": int(usage.free),
        "free_gb": round(usage.free / (1024**3), 3),
        "total_gb": round(usage.total / (1024**3), 3),
    }


def write_checkpoint_layout_report(path, repo_path=None):
    status = checkpoint_status(foundation_stereo_repo=repo_path)
    links = official_model_links(repo_path)
    lines = [
        "# FoundationStereo Checkpoint Layout",
        "",
        f"- repo_path: `{status['repo_path']}`",
        f"- repo_exists: `{status['repo_exists']}`",
        f"- readme_path: `{status['readme_path']}`",
        f"- default_model_dir: `{status['model_dir']}`",
        f"- ckpt_path: `{status['ckpt_path']}`",
        f"- cfg_path: `{status['cfg_path']}`",
        f"- checkpoint_ready: `{status['checkpoint_ready']}`",
        "",
        "## Confirmed From Local Repository",
        "",
        "- The local repo uses lowercase `readme.md`, not root `README.md`.",
        "- `readme.md` says to put the entire model folder, for example `23-51-11`, under `./pretrained_models/`.",
        "- `scripts/run_demo.py` defaults `--ckpt_dir` to `../pretrained_models/23-51-11/model_best_bp2.pth`.",
        "- `scripts/run_demo.py` loads `cfg.yaml` from the checkpoint parent directory.",
        "- Inputs are left/right rectified, undistorted stereo RGB images; left/right must not be swapped.",
        "- The demo converts disparity to depth with `depth = K[0,0] * baseline / disp`.",
        "",
        "## Official Model Links Found In README",
        "",
    ]
    if links:
        lines += [f"- `{name}`: {url}" for name, url in links.items()]
    else:
        lines.append("- None found in local README.")
    lines += [
        "",
        "## Missing Files",
        "",
    ]
    lines += [f"- `{item}`" for item in status["missing_files"]] or ["- None"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status
