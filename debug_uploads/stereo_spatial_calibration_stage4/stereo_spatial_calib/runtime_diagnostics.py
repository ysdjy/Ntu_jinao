"""Runtime diagnostics for IsaacLab and FoundationStereo environments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .foundation_stereo_adapter import find_checkpoint, find_foundation_stereo_repo


def import_status(module_name):
    try:
        mod = __import__(module_name)
        return {"available": True, "version": getattr(mod, "__version__", None)}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def current_python_report(include_sys_path=True):
    report = {
        "executable": sys.executable,
        "version": sys.version,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }
    if include_sys_path:
        report["sys_path"] = list(sys.path)
    return report


def module_runtime_report():
    modules = {}
    for module in [
        "torch",
        "cv2",
        "numpy",
        "scipy",
        "matplotlib",
        "open3d",
        "yaml",
        "imageio",
        "omegaconf",
        "isaacsim",
        "isaaclab",
        "omni",
        "isaacsim.simulation_app",
    ]:
        modules[module] = import_status(module)
    cuda = {"available": False}
    if modules["torch"]["available"]:
        import torch

        cuda["available"] = bool(torch.cuda.is_available())
        if cuda["available"]:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            cuda.update(
                {
                    "device_index": int(idx),
                    "device_name": torch.cuda.get_device_name(idx),
                    "total_memory_gb": round(props.total_memory / (1024**3), 3),
                }
            )
    return modules, cuda


def conda_env_exists(env_name="env_isaaclab"):
    conda = Path.home() / "miniconda3" / "bin" / "conda"
    if not conda.exists():
        return False
    result = subprocess.run([str(conda), "env", "list", "--json"], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return False
    try:
        envs = json.loads(result.stdout).get("envs", [])
    except json.JSONDecodeError:
        return False
    return any(Path(env).name == env_name for env in envs)


def probe_conda_env(env_name="env_isaaclab", timeout=30):
    if not conda_env_exists(env_name):
        return {"exists": False}
    conda_sh = Path.home() / "miniconda3" / "etc" / "profile.d" / "conda.sh"
    if not conda_sh.exists():
        return {"exists": True, "error": f"conda.sh not found: {conda_sh}"}
    probe = r"""
import json, os, sys
def check(name):
    try:
        mod = __import__(name)
        return {"available": True, "version": getattr(mod, "__version__", None)}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
mods = {m: check(m) for m in ["torch","isaacsim","isaaclab","omni","isaacsim.simulation_app","cv2","open3d","omegaconf","imageio","scipy"]}
cuda = {"available": False}
if mods["torch"]["available"]:
    import torch
    cuda["available"] = bool(torch.cuda.is_available())
    if cuda["available"]:
        cuda["device_name"] = torch.cuda.get_device_name(0)
print(json.dumps({
    "python_executable": sys.executable,
    "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
    "modules": mods,
    "cuda": cuda,
}))
"""
    cmd = f"source {conda_sh} && conda activate {env_name} && python - <<'PY'\n{probe}\nPY"
    try:
        result = subprocess.run(
            ["bash", "-lc", cmd],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"exists": True, "error": f"probe timed out after {timeout}s"}
    out = result.stdout.strip().splitlines()
    payload = out[-1] if out else "{}"
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {"stdout": result.stdout, "stderr": result.stderr}
    parsed["exists"] = True
    parsed["returncode"] = result.returncode
    if result.stderr.strip():
        parsed["stderr"] = result.stderr.strip()
    return parsed


def foundation_stereo_report(repo_path=None, checkpoint_path=None):
    repo = Path(repo_path) if repo_path else find_foundation_stereo_repo()
    ckpt = Path(checkpoint_path) if checkpoint_path else find_checkpoint(repo)
    expected_paths = []
    if repo is not None:
        expected_paths = [
            str(Path(repo) / "pretrained_models" / "23-51-11" / "model_best_bp2.pth"),
            str(Path(repo) / "pretrained_models" / "11-33-40" / "model_best_bp2.pth"),
        ]
    deps = {}
    for module in ["torch", "cv2", "imageio", "open3d", "omegaconf", "scipy", "numpy"]:
        deps[module] = import_status(module)
    cfg_exists = bool(ckpt and (Path(ckpt).parent / "cfg.yaml").exists())
    return {
        "repo_path": str(repo) if repo else None,
        "repo_exists": bool(repo and Path(repo).exists()),
        "checkpoint_path": str(ckpt) if ckpt else None,
        "checkpoint_exists": bool(ckpt and Path(ckpt).exists()),
        "checkpoint_cfg_exists": cfg_exists,
        "expected_checkpoint_paths": expected_paths,
        "dependencies": deps,
        "ready": bool(repo and ckpt and Path(ckpt).exists() and cfg_exists and deps["torch"]["available"]),
        "download_hint": (
            "Download a FoundationStereo model folder such as 23-51-11 from the official README link "
            "and place it under third_party/FoundationStereo/pretrained_models/23-51-11/ with "
            "model_best_bp2.pth and cfg.yaml."
        ),
    }


def build_stage2_diagnosis(repo_path=None, checkpoint_path=None):
    modules, cuda = module_runtime_report()
    env_probe = probe_conda_env("env_isaaclab")
    fs = foundation_stereo_report(repo_path, checkpoint_path)
    return {
        "current_python": current_python_report(include_sys_path=True),
        "modules": modules,
        "cuda": cuda,
        "env_isaaclab_probe": env_probe,
        "foundation_stereo": fs,
    }


def runtime_recommendation_markdown(report):
    current = report["current_python"]
    env = report.get("env_isaaclab_probe", {})
    modules = report.get("modules", {})
    fs = report.get("foundation_stereo", {})
    isaac_ok = env.get("modules", {}).get("isaaclab", {}).get("available") and env.get("modules", {}).get(
        "isaacsim", {}
    ).get("available")
    torch_ok = env.get("modules", {}).get("torch", {}).get("available")
    lines = [
        "# Stage 2 Runtime Recommendation",
        "",
        f"- Current `./isaaclab.sh -p` Python in this run: `{current.get('executable')}`",
        f"- Current conda env: `{current.get('conda_default_env')}`",
        f"- Current torch import: `{modules.get('torch', {}).get('available')}`",
        f"- Current isaacsim import: `{modules.get('isaacsim', {}).get('available')}`",
        f"- Current isaaclab import: `{modules.get('isaaclab', {}).get('available')}`",
        "",
        "## Diagnosis",
        "",
    ]
    if current.get("conda_default_env") == "base" and env.get("exists"):
        lines.append(
            "`./isaaclab.sh -p` is currently bound to the active base conda Python. "
            "That base environment does not contain torch, isaacsim, or isaaclab, so imports fail."
        )
    else:
        lines.append("The active Python should be checked against the JSON report before running native Isaac scripts.")
    lines += [
        "",
        "## Recommended Runtime",
        "",
        f"- `env_isaaclab` exists: `{env.get('exists')}`",
        f"- `env_isaaclab` Python: `{env.get('python_executable')}`",
        f"- `env_isaaclab` torch available: `{torch_ok}`",
        f"- `env_isaaclab` CUDA available: `{env.get('cuda', {}).get('available')}`",
        f"- `env_isaaclab` GPU: `{env.get('cuda', {}).get('device_name')}`",
        f"- `env_isaaclab` Isaac Sim/Lab available: `{bool(isaac_ok)}`",
        "",
        "Use:",
        "",
        "```bash",
        "source /home1/banghai/miniconda3/etc/profile.d/conda.sh",
        "conda activate env_isaaclab",
        "cd ~/IsaacLab",
        "TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/00_check_environment.py",
        "```",
        "",
        "Do not install torch into base just to fix this project. Use the existing `env_isaaclab` environment.",
        "",
        "## isaaclab.sh",
        "",
        "No code change to `isaaclab.sh` is required if the correct conda environment is activated before running it. "
        "Only consider modifying `isaaclab.sh` if you intentionally want to force a global Python path for all IsaacLab work.",
        "",
        "## FoundationStereo",
        "",
        f"- Repo: `{fs.get('repo_path')}`",
        f"- Checkpoint: `{fs.get('checkpoint_path')}`",
        f"- Checkpoint ready: `{fs.get('ready')}`",
        "",
        "Expected checkpoint locations include:",
    ]
    lines += [f"- `{p}`" for p in fs.get("expected_checkpoint_paths", [])]
    lines += [
        "",
        "Fallback command while checkpoint is missing:",
        "",
        "```bash",
        "TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/run_full_pipeline.py \\",
        "  --num_samples_per_object 1 \\",
        "  --objects cube sphere cylinder \\",
        "  --resolution 640 480 \\",
        "  --baseline 0.10 \\",
        "  --backend synthetic \\",
        "  --depth_source gt",
        "```",
        "",
    ]
    return "\n".join(lines)

