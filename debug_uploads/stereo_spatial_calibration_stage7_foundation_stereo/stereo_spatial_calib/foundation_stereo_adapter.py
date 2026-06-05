"""FoundationStereo adapter with a GT-depth fallback for geometry validation."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .camera_geometry import disparity_to_depth
from .foundation_stereo_checkpoint import (
    DEFAULT_REPO_CANDIDATES as FOUNDATION_REPO_CANDIDATES,
    checkpoint_status,
    default_model_dir,
    find_foundation_stereo_repo,
)
from .io_utils import colorize_scalar, ensure_dir, read_json, save_image, write_json


def find_checkpoint(repo_path=None):
    candidates = []
    if repo_path is not None:
        repo = Path(repo_path)
        expected = [
            repo / "pretrained_models" / "23-51-11" / "model_best_bp2.pth",
            repo / "pretrained_models" / "11-33-40" / "model_best_bp2.pth",
        ]
        for path in expected:
            if path.exists():
                return path
        candidates.extend(repo.rglob("*.pth"))
        candidates.extend(repo.rglob("*.pt"))
        candidates.extend(repo.rglob("*.ckpt"))
    for path in candidates:
        name = path.name.lower()
        if "foundation" in name or "stereo" in name or "model" in name:
            return path
    return candidates[0] if candidates else None


def expected_checkpoint_files(repo_path=None):
    root = default_model_dir(repo_path)
    return root / "model_best_bp2.pth", root / "cfg.yaml"


class FoundationStereoAdapter:
    def __init__(self, repo_path=None, checkpoint_path=None, max_depth_m=10.0, model_dir=None, cfg_path=None):
        self.repo_path = Path(repo_path).expanduser() if repo_path else find_foundation_stereo_repo()
        status = checkpoint_status(
            foundation_stereo_repo=self.repo_path,
            foundation_stereo_model_dir=model_dir,
            foundation_stereo_ckpt=checkpoint_path,
            foundation_stereo_cfg=cfg_path,
        )
        self.model_dir = Path(status["model_dir"])
        self.checkpoint_path = Path(status["ckpt_path"])
        self.cfg_path = Path(status["cfg_path"])
        self.max_depth_m = float(max_depth_m)

    @property
    def ready(self):
        return bool(
            self.repo_path is not None
            and self.repo_path.exists()
            and self.checkpoint_path is not None
            and self.checkpoint_path.exists()
            and self.cfg_path.exists()
            and self.torch_available()
        )

    @staticmethod
    def torch_available():
        try:
            import torch  # noqa: F401

            return True
        except Exception:
            return False

    def status(self):
        status = checkpoint_status(
            foundation_stereo_repo=self.repo_path,
            foundation_stereo_model_dir=self.model_dir,
            foundation_stereo_ckpt=self.checkpoint_path,
            foundation_stereo_cfg=self.cfg_path,
        )
        status.update(
            {
            "repo_path": str(self.repo_path) if self.repo_path else None,
            "model_dir": str(self.model_dir),
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "cfg_path": str(self.cfg_path) if self.cfg_path else None,
            "expected_checkpoint_path": str(self.model_dir / "model_best_bp2.pth"),
            "expected_cfg_path": str(self.model_dir / "cfg.yaml"),
            "checkpoint_exists": bool(self.checkpoint_path and self.checkpoint_path.exists()),
            "checkpoint_cfg_exists": bool(self.cfg_path and self.cfg_path.exists()),
            "checkpoint_ready": bool(self.ready),
            "torch_available": self.torch_available(),
            "ready": bool(self.ready),
            "download_hint": (
                "Download a FoundationStereo model folder such as 23-51-11 from the official README link "
                "and place it under third_party/FoundationStereo/pretrained_models/23-51-11/ with "
                "model_best_bp2.pth and cfg.yaml."
            ),
            "fallback_command": (
                "TERM=xterm ./isaaclab.sh -p "
                "source/standalone/stereo_spatial_calibration/scripts/run_full_pipeline.py "
                "--backend synthetic --depth_source gt"
            ),
            }
        )
        return status

    def infer_disparity(self, left_rgb_path, right_rgb_path, output_dir):
        """Run FoundationStereo's PyTorch model and return pixel disparity."""

        if not self.ready:
            raise RuntimeError(f"FoundationStereo is not ready: {self.status()}")
        out = ensure_dir(output_dir)
        start = time.perf_counter()
        repo = str(self.repo_path)
        old_path = list(sys.path)
        try:
            if repo not in sys.path:
                sys.path.insert(0, repo)
            import cv2
            import imageio.v2 as imageio
            import torch
            from omegaconf import OmegaConf

            from core.foundation_stereo import FoundationStereo
            from core.utils.utils import InputPadder

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cfg = OmegaConf.load(str(self.cfg_path))
            if "vit_size" not in cfg:
                cfg["vit_size"] = "vitl"
            cfg.ckpt_dir = str(self.checkpoint_path)
            cfg.valid_iters = int(getattr(cfg, "valid_iters", 32))
            cfg.hiera = int(getattr(cfg, "hiera", 0))
            model = FoundationStereo(cfg)
            # PyTorch 2.6+ defaults torch.load(weights_only=True), but the
            # official FoundationStereo checkpoint contains numpy scalar
            # metadata that requires trusted full checkpoint loading.
            ckpt = torch.load(str(self.checkpoint_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            model.to(device)
            model.eval()
            img0 = imageio.imread(str(left_rgb_path))
            img1 = imageio.imread(str(right_rgb_path))
            if img0.shape[:2] != img1.shape[:2]:
                raise ValueError(f"left/right image shape mismatch: {img0.shape} vs {img1.shape}")
            if img0.ndim != 3 or img0.shape[2] != 3:
                raise ValueError(f"FoundationStereo expects RGB images, got {img0.shape}")
            img0 = np.asarray(img0)
            img1 = np.asarray(img1)
            H, W = img0.shape[:2]
            img0_t = torch.as_tensor(img0).to(device).float()[None].permute(0, 3, 1, 2)
            img1_t = torch.as_tensor(img1).to(device).float()[None].permute(0, 3, 1, 2)
            padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
            img0_t, img1_t = padder.pad(img0_t, img1_t)
            amp_enabled = bool(device.type == "cuda")
            with torch.no_grad():
                with torch.cuda.amp.autocast(amp_enabled):
                    if not int(getattr(cfg, "hiera", 0)):
                        disp = model.forward(img0_t, img1_t, iters=int(getattr(cfg, "valid_iters", 32)), test_mode=True)
                    else:
                        disp = model.run_hierachical(
                            img0_t,
                            img1_t,
                            iters=int(getattr(cfg, "valid_iters", 32)),
                            test_mode=True,
                            small_ratio=0.5,
                        )
            disp = padder.unpad(disp.float()).detach().cpu().numpy().reshape(H, W).astype(np.float32)
            runtime_ms = (time.perf_counter() - start) * 1000.0
            np.save(out / "foundation_stereo_raw_disparity.npy", disp)
            write_json(
                out / "foundation_stereo_runtime.json",
                {
                    "status": "success",
                    "runtime_ms": runtime_ms,
                    "repo_path": str(self.repo_path),
                    "model_dir": str(self.model_dir),
                    "checkpoint_path": str(self.checkpoint_path),
                    "cfg_path": str(self.cfg_path),
                    "device": str(device),
                    "cuda_available": bool(device.type == "cuda"),
                },
            )
            return disp, runtime_ms
        finally:
            sys.path = old_path

    def _gt_depth_prediction(self, sample_path, intr, baseline, mode):
        pred_depth = np.load(sample_path / "gt_depth_left.npy").astype(np.float32)
        disparity = np.full(pred_depth.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(pred_depth) & (pred_depth > 0.0)
        disparity[valid] = float(intr["fx"]) * baseline / pred_depth[valid]
        return disparity, pred_depth, mode, 0.0

    def _noisy_gt_depth_prediction(self, sample_path, intr, baseline, std_m, bias_m, seed=0):
        gt_depth = np.load(sample_path / "gt_depth_left.npy").astype(np.float32)
        rng = np.random.default_rng(seed)
        noise = rng.normal(float(bias_m), float(std_m), size=gt_depth.shape).astype(np.float32)
        pred_depth = gt_depth.copy()
        valid = np.isfinite(gt_depth) & (gt_depth > 0.0)
        pred_depth[valid] = np.maximum(gt_depth[valid] + noise[valid], 1e-6)
        disparity = np.full(pred_depth.shape, np.nan, dtype=np.float32)
        disparity[valid] = float(intr["fx"]) * baseline / pred_depth[valid]
        return disparity, pred_depth, "noisy_gt_depth", 0.0

    def process_sample(
        self,
        sample_dir,
        predictions_root,
        use_gt_depth_as_prediction=False,
        depth_source=None,
        depth_noise_std_m=0.005,
        depth_noise_bias_m=0.0,
        allow_fallback=True,
    ):
        sample_path = Path(sample_dir)
        metadata = read_json(sample_path / "metadata.json")
        sample_id = metadata["sample_id"]
        pred_dir = ensure_dir(Path(predictions_root) / sample_id)
        intr = metadata["left_camera_intrinsics"]
        baseline = float(metadata["baseline_m"])

        if depth_source is None:
            depth_source = "gt" if use_gt_depth_as_prediction else "foundation_stereo"

        runtime_status = {
            "sample_id": sample_id,
            "requested_depth_source": depth_source,
            "status": "unknown",
            "input_left_path": str(sample_path / "left_rgb.png"),
            "input_right_path": str(sample_path / "right_rgb.png"),
            "fx": float(intr["fx"]),
            "baseline_m": float(baseline),
        }
        if depth_source == "gt":
            disparity, pred_depth, mode, runtime_ms = self._gt_depth_prediction(sample_path, intr, baseline, "gt_depth")
            runtime_status.update({"status": "success", "mode": mode, "runtime_ms": runtime_ms})
        elif depth_source == "noisy_gt":
            seed = sum((idx + 1) * byte for idx, byte in enumerate(sample_id.encode("utf-8"))) % (2**32)
            disparity, pred_depth, mode, runtime_ms = self._noisy_gt_depth_prediction(
                sample_path, intr, baseline, depth_noise_std_m, depth_noise_bias_m, seed=seed
            )
            runtime_status.update(
                {
                    "status": "success",
                    "mode": mode,
                    "runtime_ms": runtime_ms,
                    "depth_noise_std_m": float(depth_noise_std_m),
                    "depth_noise_bias_m": float(depth_noise_bias_m),
                }
            )
        elif depth_source == "foundation_stereo":
            if not self.ready:
                runtime_status.update(
                    {
                        "status": "checkpoint_missing",
                        "mode": "foundation_stereo_checkpoint_missing",
                        "runtime_ms": 0.0,
                        "adapter_status": self.status(),
                    }
                )
                self._finalize_runtime(pred_dir, runtime_status, disparity=None, pred_depth=None)
                return {
                    "sample_id": sample_id,
                    "mode": "foundation_stereo_checkpoint_missing",
                    "requested_depth_source": depth_source,
                    "actual_depth_source": "foundation_stereo_checkpoint_missing",
                    "foundation_stereo_runtime_ms": 0.0,
                    "prediction_dir": str(pred_dir),
                    "skipped": True,
                    "skip_reason": "checkpoint_missing",
                }
            try:
                disparity, runtime_ms = self.infer_disparity(
                    sample_path / "left_rgb.png", sample_path / "right_rgb.png", pred_dir
                )
                pred_depth = disparity_to_depth(disparity, intr["fx"], baseline, max_depth_m=self.max_depth_m)
                mode = "foundation_stereo_depth"
                runtime_status.update({"status": "success", "mode": mode, "runtime_ms": runtime_ms})
            except Exception as exc:
                runtime_status.update(
                    {
                        "status": "inference_failed",
                        "mode": "foundation_stereo_inference_failed",
                        "runtime_ms": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                        "adapter_status": self.status(),
                    }
                )
                self._finalize_runtime(pred_dir, runtime_status, disparity=None, pred_depth=None)
                if not allow_fallback:
                    raise
                return {
                    "sample_id": sample_id,
                    "mode": "foundation_stereo_inference_failed",
                    "requested_depth_source": depth_source,
                    "actual_depth_source": "foundation_stereo_inference_failed",
                    "foundation_stereo_runtime_ms": 0.0,
                    "prediction_dir": str(pred_dir),
                    "skipped": True,
                    "skip_reason": "inference_failed",
                }
        else:
            raise ValueError(f"unknown depth_source: {depth_source}")

        np.save(pred_dir / "disparity.npy", disparity.astype(np.float32))
        np.save(pred_dir / "pred_depth.npy", pred_depth.astype(np.float32))
        save_image(pred_dir / "disparity_color.png", colorize_scalar(disparity))
        save_image(pred_dir / "depth_color.png", colorize_scalar(pred_depth))
        save_image(pred_dir / "pred_depth_color.png", colorize_scalar(pred_depth))
        runtime_status.setdefault("mode", mode)
        runtime_status.setdefault("runtime_ms", runtime_ms)
        runtime_status["actual_depth_source"] = mode
        runtime_status["prediction_dir"] = str(pred_dir)
        self._finalize_runtime(pred_dir, runtime_status, disparity=disparity, pred_depth=pred_depth)
        return {
            "sample_id": sample_id,
            "mode": mode,
            "requested_depth_source": depth_source,
            "actual_depth_source": mode,
            "foundation_stereo_runtime_ms": float(runtime_status.get("runtime_ms", 0.0)),
            "prediction_dir": str(pred_dir),
            "skipped": False,
        }

    def _finalize_runtime(self, pred_dir, runtime_status, disparity=None, pred_depth=None):
        pred_dir = ensure_dir(pred_dir)
        disp_valid = np.isfinite(disparity) & (disparity > 0.0) if disparity is not None else np.array([], dtype=bool)
        depth_valid = np.isfinite(pred_depth) & (pred_depth > 0.0) if pred_depth is not None else np.array([], dtype=bool)
        status = self.status()
        mode = runtime_status.get("mode", "")
        requested_depth_source = runtime_status.get("requested_depth_source", "")
        runtime_status.update(
            {
                "foundation_stereo_repo": str(self.repo_path) if self.repo_path else None,
                "model_dir": str(self.model_dir),
                "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
                "cfg_path": str(self.cfg_path) if self.cfg_path else None,
                "checkpoint_available": bool(self.ready),
                "checkpoint_ready": bool(self.ready),
                "missing_files": status.get("missing_files", []),
                "inference_success": bool(
                    requested_depth_source == "foundation_stereo" and mode == "foundation_stereo_depth"
                ),
                "disparity_min": float(np.nanmin(disparity[disp_valid])) if disp_valid.any() else None,
                "disparity_max": float(np.nanmax(disparity[disp_valid])) if disp_valid.any() else None,
                "disparity_mean": float(np.nanmean(disparity[disp_valid])) if disp_valid.any() else None,
                "pred_depth_valid_ratio": float(depth_valid.sum() / pred_depth.size) if pred_depth is not None and pred_depth.size else 0.0,
                "pred_depth_min": float(np.nanmin(pred_depth[depth_valid])) if depth_valid.any() else None,
                "pred_depth_max": float(np.nanmax(pred_depth[depth_valid])) if depth_valid.any() else None,
                "pred_depth_mean": float(np.nanmean(pred_depth[depth_valid])) if depth_valid.any() else None,
            }
        )
        write_json(pred_dir / "foundation_stereo_runtime.json", runtime_status)


def clone_foundation_stereo(dest=None):
    dest = Path(dest) if dest else FOUNDATION_REPO_CANDIDATES[-1]
    if dest.exists():
        return dest, "exists"
    ensure_dir(dest.parent)
    git = shutil.which("git")
    if not git:
        return None, "git_not_found"
    try:
        result = subprocess.run(
            [git, "clone", "https://github.com/NVlabs/FoundationStereo.git", str(dest)],
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None, "clone_failed: timeout_after_120s"
    if result.returncode != 0:
        return None, f"clone_failed: {result.stderr.strip()}"
    return dest, "cloned"
