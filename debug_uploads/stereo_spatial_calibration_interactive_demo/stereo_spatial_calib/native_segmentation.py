"""Helpers for Isaac native segmentation and mask-source selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


MASK_SOURCE_ORDER = [
    "native_instance",
    "native_instance_id",
    "native_semantic",
    "native_bbox_2d_tight",
    "projected_bbox_fallback",
]

MASK_SOURCE_ALIASES = {
    "auto": MASK_SOURCE_ORDER,
    "native_instance": ["native_instance"],
    "native_instance_id": ["native_instance_id"],
    "native_semantic": ["native_semantic"],
    "native_bbox_2d_tight": ["native_bbox_2d_tight"],
    "projected_bbox_fallback": ["projected_bbox_fallback"],
}

ANNOTATOR_BY_SOURCE = {
    "native_instance": ("add_instance_segmentation_to_frame", "instance_segmentation"),
    "native_instance_id": ("add_instance_id_segmentation_to_frame", "instance_id_segmentation"),
    "native_semantic": ("add_semantic_segmentation_to_frame", "semantic_segmentation"),
    "native_bbox_2d_tight": ("add_bounding_box_2d_tight_to_frame", "bounding_box_2d_tight"),
}


@dataclass
class MaskSelection:
    mask: np.ndarray | None
    mask_source: str
    segmentation_available: bool
    target_instance_id: int | None = None
    frame_summary: dict | None = None
    raw_segmentation: np.ndarray | None = None
    colorized_segmentation: np.ndarray | None = None
    bbox: list[int] | None = None


def ordered_mask_sources(mask_source="auto"):
    if mask_source not in MASK_SOURCE_ALIASES:
        raise ValueError(f"unknown mask_source: {mask_source}")
    return list(MASK_SOURCE_ALIASES[mask_source])


def attach_segmentation_annotators(camera, mask_source="auto"):
    attached = []
    for source in ordered_mask_sources(mask_source):
        spec = ANNOTATOR_BY_SOURCE.get(source)
        if spec is None:
            continue
        add_name, key = spec
        if hasattr(camera, add_name):
            try:
                getattr(camera, add_name)()
                attached.append({"mask_source": source, "annotator": key, "status": "attached"})
            except Exception as exc:
                attached.append({"mask_source": source, "annotator": key, "status": f"failed: {exc!r}"})
        else:
            attached.append({"mask_source": source, "annotator": key, "status": "method_missing"})
    return attached


def _to_numpy(data):
    if data is None:
        return None
    try:
        import torch

        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import warp as wp

        if isinstance(data, wp.array):
            data = data.numpy()
    except Exception:
        pass
    arr = np.asarray(data)
    return arr if arr.size else None


def summarize_frame(frame: dict[str, Any]):
    summary = {}
    for key, value in (frame or {}).items():
        item = {}
        data = value.get("data") if isinstance(value, dict) else value
        arr = _to_numpy(data)
        if arr is not None:
            item["shape"] = list(arr.shape)
            item["dtype"] = str(arr.dtype)
            if np.issubdtype(arr.dtype, np.number):
                finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.floating) else np.ones(arr.shape, dtype=bool)
                if finite.any():
                    item["min"] = float(np.nanmin(arr[finite]))
                    item["max"] = float(np.nanmax(arr[finite]))
                if arr.size <= 1_000_000:
                    item["unique_count"] = int(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0).shape[0]) if arr.ndim == 3 else int(np.unique(arr).size)
        if isinstance(value, dict):
            for info_key in ("info", "idToLabels", "idToSemantics", "idToLabelsMap"):
                if info_key in value:
                    item[info_key] = _shorten(value[info_key])
        summary[key] = item
    return summary


def _shorten(value, limit=50):
    if isinstance(value, dict):
        out = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= limit:
                out["..."] = f"{len(value) - limit} more"
                break
            out[str(k)] = str(v)[:300]
        return out
    return str(value)[:1000]


def is_colorized_segmentation(arr):
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return False
    if arr.dtype == np.uint8:
        return True
    if arr.shape[-1] in (3, 4) and np.unique(arr.reshape(-1, arr.shape[-1]), axis=0).shape[0] > 8:
        return True
    return False


def extract_id_map(frame_value):
    raw = frame_value.get("data") if isinstance(frame_value, dict) else frame_value
    arr = _to_numpy(raw)
    if arr is None:
        return None
    if is_colorized_segmentation(arr):
        return None
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None
    if not np.issubdtype(arr.dtype, np.integer) and not np.all(np.equal(arr, np.floor(arr)) | ~np.isfinite(arr)):
        return None
    return arr.astype(np.int64)


def extract_colorized(frame_value):
    raw = frame_value.get("data") if isinstance(frame_value, dict) else frame_value
    arr = _to_numpy(raw)
    if arr is not None and is_colorized_segmentation(arr):
        return np.asarray(arr[..., :3], dtype=np.uint8)
    return None


def _labels_from_value(frame_value):
    if not isinstance(frame_value, dict):
        return {}
    out = {}
    for key in ("idToLabels", "idToSemantics"):
        value = frame_value.get(key)
        if isinstance(value, dict):
            out.update(value)
    info = frame_value.get("info")
    if isinstance(info, dict):
        for key in ("idToLabels", "idToSemantics"):
            value = info.get(key)
            if isinstance(value, dict):
                out.update(value)
    return out


def select_target_id(id_map, labels=None, target_names=()):
    arr = np.asarray(id_map, dtype=np.int64)
    positive = arr[arr > 0]
    if positive.size == 0:
        return None
    labels = labels or {}
    names = [str(x).lower() for x in target_names if x]
    for raw_id, label in labels.items():
        try:
            idx = int(raw_id)
        except Exception:
            continue
        text = str(label).lower()
        if any(name in text for name in names) and np.any(arr == idx):
            return idx
    ids, counts = np.unique(positive, return_counts=True)
    return int(ids[np.argmax(counts)])


def mask_from_id_map(id_map, target_id):
    if target_id is None:
        return None
    mask = np.asarray(id_map, dtype=np.int64) == int(target_id)
    return mask if mask.any() else None


def bbox_to_mask(bbox, shape, shrink_ratio=0.9):
    if bbox is None:
        return None
    x0, y0, x1, y1 = [float(x) for x in bbox]
    h, w = int(shape[0]), int(shape[1])
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    hw, hh = 0.5 * (x1 - x0) * float(shrink_ratio), 0.5 * (y1 - y0) * float(shrink_ratio)
    x0 = int(np.clip(np.floor(cx - hw), 0, w - 1))
    x1 = int(np.clip(np.ceil(cx + hw), 0, w - 1))
    y0 = int(np.clip(np.floor(cy - hh), 0, h - 1))
    y1 = int(np.clip(np.ceil(cy + hh), 0, h - 1))
    if x1 < x0 or y1 < y0:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def extract_bbox(frame_value, target_names=()):
    raw = frame_value.get("data") if isinstance(frame_value, dict) else frame_value
    arr = _to_numpy(raw)
    if arr is None or arr.size == 0:
        return None
    arr = np.asarray(arr)
    candidates = []
    if arr.dtype.fields:
        for item in arr.reshape(-1):
            fields = item.dtype.fields
            names = {name.lower(): name for name in fields}
            required = ["x_min", "y_min", "x_max", "y_max"]
            if all(k in names for k in required):
                bbox = [int(item[names[k]]) for k in required]
                area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
                if area > 0:
                    candidates.append((area, bbox))
    elif arr.ndim >= 2 and arr.shape[-1] >= 4:
        flat = arr.reshape(-1, arr.shape[-1])
        for row in flat:
            bbox = [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
            area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
            if area > 0:
                candidates.append((area, bbox))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def select_mask_from_frame(frame, shape, mask_source="auto", target_names=()):
    frame = frame or {}
    summary = summarize_frame(frame)
    for source in ordered_mask_sources(mask_source):
        if source == "projected_bbox_fallback":
            continue
        _, key = ANNOTATOR_BY_SOURCE[source]
        if key not in frame:
            continue
        if source == "native_bbox_2d_tight":
            bbox = extract_bbox(frame[key], target_names=target_names)
            mask = bbox_to_mask(bbox, shape)
            if mask is not None and mask.any():
                return MaskSelection(mask, source, True, frame_summary=summary, bbox=bbox)
            continue
        id_map = extract_id_map(frame[key])
        colorized = extract_colorized(frame[key])
        if id_map is None:
            continue
        target_id = select_target_id(id_map, labels=_labels_from_value(frame[key]), target_names=target_names)
        mask = mask_from_id_map(id_map, target_id)
        if mask is not None and mask.any():
            return MaskSelection(mask, source, True, target_id, summary, id_map, colorized)
    return MaskSelection(None, "projected_bbox_fallback", False, frame_summary=summary)
