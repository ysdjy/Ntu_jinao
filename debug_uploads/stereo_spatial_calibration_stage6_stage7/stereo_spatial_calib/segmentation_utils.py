"""Utilities for Isaac GT segmentation / instance mask handling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_object_mask(sample_dir, object_name=None, object_id=None):
    """Load the object mask for one sample.

    First version expects ``gt_instance_mask.npy`` to be a boolean or integer mask where
    target pixels are non-zero. If a multi-id segmentation is provided, pass object_id.
    """

    sample_path = Path(sample_dir)
    mask_path = sample_path / "gt_instance_mask.npy"
    if not mask_path.exists():
        mask_path = sample_path / "gt_segmentation.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"no mask found in {sample_path}")

    mask = np.load(mask_path)
    if object_id is not None:
        return mask == int(object_id)
    if mask.dtype == np.bool_:
        return mask
    if object_name is not None and (sample_path / "metadata.json").exists():
        with open(sample_path / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        name_to_id = metadata.get("instance_name_to_id", {})
        if object_name in name_to_id:
            return mask == int(name_to_id[object_name])
    return mask > 0


def mask_to_bbox(mask):
    mask_arr = np.asarray(mask).astype(bool)
    ys, xs = np.nonzero(mask_arr)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def get_mask_center_pixel(mask):
    mask_arr = np.asarray(mask).astype(bool)
    ys, xs = np.nonzero(mask_arr)
    if len(xs) == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def erode_mask(mask, kernel_size=3):
    mask_arr = np.asarray(mask).astype(bool)
    if kernel_size <= 1:
        return mask_arr
    try:
        import cv2

        kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
        return cv2.erode(mask_arr.astype(np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        pad = int(kernel_size) // 2
        padded = np.pad(mask_arr, pad, mode="constant", constant_values=False)
        out = np.zeros_like(mask_arr, dtype=bool)
        for y in range(mask_arr.shape[0]):
            for x in range(mask_arr.shape[1]):
                window = padded[y : y + kernel_size, x : x + kernel_size]
                out[y, x] = bool(window.all())
        return out


def apply_mask_shift(mask, shift_x_px=0, shift_y_px=0):
    """Shift a mask by integer pixels, filling exposed regions with false."""

    src = np.asarray(mask).astype(bool)
    out = np.zeros_like(src, dtype=bool)
    sx = int(shift_x_px)
    sy = int(shift_y_px)
    h, w = src.shape
    src_x0 = max(0, -sx)
    src_x1 = min(w, w - sx)
    dst_x0 = max(0, sx)
    dst_x1 = min(w, w + sx)
    src_y0 = max(0, -sy)
    src_y1 = min(h, h - sy)
    dst_y0 = max(0, sy)
    dst_y1 = min(h, h + sy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = src[src_y0:src_y1, src_x0:src_x1]
    return out


def dilate_mask(mask, kernel_size=3):
    mask_arr = np.asarray(mask).astype(bool)
    if kernel_size <= 1:
        return mask_arr
    try:
        import cv2

        kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
        return cv2.dilate(mask_arr.astype(np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        pad = int(kernel_size) // 2
        padded = np.pad(mask_arr, pad, mode="constant", constant_values=False)
        out = np.zeros_like(mask_arr, dtype=bool)
        for y in range(mask_arr.shape[0]):
            for x in range(mask_arr.shape[1]):
                window = padded[y : y + kernel_size, x : x + kernel_size]
                out[y, x] = bool(window.any())
        return out


def apply_mask_morphology(mask, mode="none", kernel_size=3):
    if mode == "none" or mode is None:
        return np.asarray(mask).astype(bool)
    if mode == "erode":
        return erode_mask(mask, kernel_size=kernel_size)
    if mode == "dilate":
        return dilate_mask(mask, kernel_size=kernel_size)
    raise ValueError(f"unknown mask morphology mode: {mode}")
