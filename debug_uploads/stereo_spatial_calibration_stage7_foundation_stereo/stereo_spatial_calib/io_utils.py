"""Small IO helpers shared by scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def sample_dirs(dataset_root):
    root = Path(dataset_root)
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "metadata.json").exists()])


def save_image(path, image_rgb):
    path = Path(path)
    ensure_dir(path.parent)
    arr = np.asarray(image_rgb)
    try:
        import cv2

        if arr.ndim == 3 and arr.shape[2] == 3:
            cv2.imwrite(str(path), arr[:, :, ::-1])
        else:
            cv2.imwrite(str(path), arr)
        return
    except Exception:
        pass
    try:
        from PIL import Image

        Image.fromarray(arr.astype(np.uint8)).save(path)
        return
    except Exception:
        pass
    import matplotlib.pyplot as plt

    plt.imsave(path, arr)


def load_image_rgb(path):
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return img[:, :, ::-1]
    except Exception:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))


def colorize_scalar(values, vmin=None, vmax=None, invalid_color=(0, 0, 0)):
    arr = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(arr)
    if vmin is None:
        vmin = float(np.nanpercentile(arr[valid], 2)) if valid.any() else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(arr[valid], 98)) if valid.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    try:
        import matplotlib.cm as cm

        rgb = (cm.get_cmap("turbo")(norm)[..., :3] * 255).astype(np.uint8)
    except Exception:
        rgb = np.stack([(norm * 255), ((1 - norm) * 255), np.zeros_like(norm)], axis=-1).astype(np.uint8)
    rgb[~valid] = np.asarray(invalid_color, dtype=np.uint8)
    return rgb
