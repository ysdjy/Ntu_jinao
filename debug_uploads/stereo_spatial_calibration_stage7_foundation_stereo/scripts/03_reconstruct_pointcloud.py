#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_spatial_calib.io_utils import OUTPUT_ROOT, read_json, sample_dirs  # noqa: E402
from stereo_spatial_calib.pointcloud_utils import reconstruct_depth_products, save_pointcloud_npy, save_pointcloud_ply  # noqa: E402
from stereo_spatial_calib.segmentation_utils import erode_mask, load_object_mask  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=str(OUTPUT_ROOT / "dataset"))
    parser.add_argument("--predictions_dir", default=str(OUTPUT_ROOT / "predictions"))
    parser.add_argument("--pointcloud_dir", default=str(OUTPUT_ROOT / "pointclouds"))
    parser.add_argument("--erode_kernel_size", type=int, default=3)
    args = parser.parse_args()

    out = Path(args.pointcloud_dir)
    count = 0
    for sample_dir in sample_dirs(args.dataset_dir):
        metadata = read_json(sample_dir / "metadata.json")
        sid = metadata["sample_id"]
        intr = metadata["left_camera_intrinsics"]
        T_world_cam = np.asarray(metadata["T_world_left_cam"], dtype=np.float32)
        T_base_world = np.asarray(metadata["T_base_world"], dtype=np.float32)
        pred_depth = np.load(Path(args.predictions_dir) / sid / "pred_depth.npy")
        gt_depth = np.load(sample_dir / "gt_depth_left.npy")
        mask = erode_mask(load_object_mask(sample_dir), args.erode_kernel_size)

        pred_cam, pred_world, pred_base = reconstruct_depth_products(pred_depth, intr, T_world_cam, T_base_world)
        gt_cam, gt_world, _ = reconstruct_depth_products(gt_depth, intr, T_world_cam, T_base_world)
        obj_cam, obj_world, _ = reconstruct_depth_products(pred_depth, intr, T_world_cam, T_base_world, mask=mask)

        save_pointcloud_npy(out / f"{sid}_pred_camera.npy", pred_cam)
        save_pointcloud_npy(out / f"{sid}_pred_world.npy", pred_world)
        save_pointcloud_npy(out / f"{sid}_pred_base.npy", pred_base)
        save_pointcloud_npy(out / f"{sid}_object_pred_world.npy", obj_world)
        save_pointcloud_ply(out / f"{sid}_pred_camera.ply", pred_cam)
        save_pointcloud_ply(out / f"{sid}_pred_world.ply", pred_world)
        save_pointcloud_ply(out / f"{sid}_object_pred_world.ply", obj_world)
        save_pointcloud_ply(out / f"{sid}_gt_world.ply", gt_world)
        count += 1
    print(f"Reconstructed pointclouds for {count} samples under {out}")


if __name__ == "__main__":
    main()

