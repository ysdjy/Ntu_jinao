"""Visualization output helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .camera_geometry import project_points
from .io_utils import colorize_scalar, ensure_dir, load_image_rgb, save_image
from .segmentation_utils import load_object_mask


def overlay_mask(image_rgb, mask, color=(255, 64, 64), alpha=0.45):
    img = np.asarray(image_rgb, dtype=np.uint8).copy()
    mask_arr = np.asarray(mask).astype(bool)
    col = np.asarray(color, dtype=np.float32)
    img[mask_arr] = (img[mask_arr].astype(np.float32) * (1.0 - alpha) + col * alpha).astype(np.uint8)
    return img


def draw_cross(image_rgb, u, v, color=(0, 255, 0), size=8):
    img = np.asarray(image_rgb, dtype=np.uint8).copy()
    x = int(round(float(u)))
    y = int(round(float(v)))
    h, w = img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return img
    x0, x1 = max(0, x - size), min(w, x + size + 1)
    y0, y1 = max(0, y - size), min(h, y + size + 1)
    img[y, x0:x1] = color
    img[y0:y1, x] = color
    return img


def write_sample_visualizations(sample_dir, prediction_dir, report_row, out_dir):
    sample_path = Path(sample_dir)
    sample_id = report_row["sample_id"]
    pred_path = Path(prediction_dir) / sample_id
    out = ensure_dir(Path(out_dir))
    rgb = load_image_rgb(sample_path / "left_rgb.png")
    mask = load_object_mask(sample_path)
    pred_depth = np.load(pred_path / "pred_depth.npy")
    gt_depth = np.load(sample_path / "gt_depth_left.npy")
    disparity_path = pred_path / "disparity.npy"
    disparity = np.load(disparity_path) if disparity_path.exists() else np.zeros_like(pred_depth)

    save_image(out / f"{sample_id}_left_rgb_with_mask.png", overlay_mask(rgb, mask))
    save_image(out / f"{sample_id}_disparity_color.png", colorize_scalar(disparity))
    save_image(out / f"{sample_id}_pred_depth_color.png", colorize_scalar(pred_depth))
    save_image(out / f"{sample_id}_gt_depth_color.png", colorize_scalar(gt_depth))
    save_image(out / f"{sample_id}_depth_error_color.png", colorize_scalar(np.abs(pred_depth - gt_depth)))

    intr = {
        "fx": report_row.get("fx", 1.0),
        "fy": report_row.get("fy", 1.0),
        "cx": report_row.get("cx", 0.0),
        "cy": report_row.get("cy", 0.0),
        "width": rgb.shape[1],
        "height": rgb.shape[0],
    }
    try:
        import json

        with open(sample_path / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        intr = metadata["left_camera_intrinsics"]
    except Exception:
        pass

    img = overlay_mask(rgb, mask, alpha=0.25)
    pred_px = project_points(np.asarray(report_row["center_camera_m"], dtype=np.float32).reshape(1, 3), intr)[0]
    img = draw_cross(img, pred_px[0], pred_px[1], color=(0, 255, 0), size=10)
    save_image(out / f"{sample_id}_object_center_overlay.png", img)

