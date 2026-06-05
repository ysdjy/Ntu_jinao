"""Error sensitivity experiments for stereo spatial localization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .camera_geometry import apply_depth_noise, perturb_baseline, perturb_intrinsics
from .coordinate_transform import perturb_transform
from .evaluator import evaluate_arrays
from .io_utils import colorize_scalar, ensure_dir, read_json, sample_dirs, save_image, write_csv, write_json
from .segmentation_utils import apply_mask_morphology, apply_mask_shift, load_object_mask


REPORT_FIELDS = [
    "sample_id",
    "object_type",
    "backend",
    "depth_source",
    "perturbation_type",
    "perturbation_value",
    "center_pred_world_m",
    "center_gt_world_m",
    "error_x_m",
    "error_y_m",
    "error_z_m",
    "error_l2_mm",
    "depth_rmse_m",
    "mask_point_count",
    "valid_depth_ratio",
]


def _stable_seed(base_seed, sample_id, label):
    text = f"{base_seed}:{sample_id}:{label}"
    return sum((idx + 1) * byte for idx, byte in enumerate(text.encode("utf-8"))) % (2**32)


def _plot_series(rows, perturbation_prefix, out_path, title):
    selected = [r for r in rows if str(r["perturbation_type"]).startswith(perturbation_prefix)]
    if not selected:
        return
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4.8))
        for key in sorted({r["perturbation_type"] for r in selected}):
            group = [r for r in selected if r["perturbation_type"] == key]
            xs = [float(str(r["perturbation_value"]).split(",")[0]) for r in group]
            ys = [float(r["error_l2_mm"]) for r in group]
            order = np.argsort(xs)
            plt.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=key)
        plt.title(title)
        plt.xlabel("perturbation value")
        plt.ylabel("object center error L2 (mm)")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return
    except Exception:
        pass

    # Minimal fallback image so the artifact always exists.
    canvas = np.full((360, 640, 3), 255, dtype=np.uint8)
    values = np.array([float(r["error_l2_mm"]) for r in selected], dtype=np.float32)
    bar = colorize_scalar(values.reshape(1, -1), vmin=float(values.min()), vmax=float(values.max()))
    canvas[170:190, 20 : 20 + bar.shape[1]] = bar[0:1]
    save_image(out_path, canvas)


def run_error_sensitivity(
    dataset_root,
    reports_dir,
    visualizations_dir,
    depth_noise_std_list=(0.0, 0.002, 0.005, 0.010),
    depth_bias_list=(-0.010, 0.0, 0.010),
    mask_shift_px_list=(-5, 0, 5),
    mask_morphology_list=("none", "erode", "dilate"),
    mask_kernel_size_list=(3, 5, 7),
    cam_trans_noise_std_list=(0.0, 0.002, 0.005),
    cam_rot_noise_deg_list=(0.0, 0.5, 1.0),
    fx_relative_error_list=(0.0, 0.005, 0.010),
    fy_relative_error_list=None,
    cx_shift_px_list=(0.0, 2.0),
    cy_shift_px_list=(0.0, 2.0),
    baseline_relative_error_list=(0.0, 0.005, 0.010),
    seed=42,
):
    rows = []
    fy_relative_error_list = fy_relative_error_list if fy_relative_error_list is not None else fx_relative_error_list
    for sample_dir in sample_dirs(dataset_root):
        metadata = read_json(sample_dir / "metadata.json")
        gt_depth = np.load(sample_dir / "gt_depth_left.npy").astype(np.float32)
        base_mask = load_object_mask(sample_dir, object_name=metadata.get("object_name"))
        intr = metadata["left_camera_intrinsics"]
        T_world_cam = np.asarray(metadata["T_world_left_cam"], dtype=np.float32)

        for std in depth_noise_std_list:
            pred = apply_depth_noise(gt_depth, std_m=float(std), bias_m=0.0, seed=_stable_seed(seed, metadata["sample_id"], f"depth_std_{std}"))
            rows.append(evaluate_arrays(metadata, pred, gt_depth, base_mask, intrinsics=intr, T_world_cam=T_world_cam, perturbation_type="depth_noise_std_m", perturbation_value=std))

        for bias in depth_bias_list:
            pred = apply_depth_noise(gt_depth, std_m=0.0, bias_m=float(bias), seed=seed)
            rows.append(evaluate_arrays(metadata, pred, gt_depth, base_mask, intrinsics=intr, T_world_cam=T_world_cam, perturbation_type="depth_bias_m", perturbation_value=bias))

        for shift in mask_shift_px_list:
            shifted = apply_mask_shift(base_mask, shift_x_px=int(shift), shift_y_px=0)
            rows.append(evaluate_arrays(metadata, gt_depth, gt_depth, shifted, intrinsics=intr, T_world_cam=T_world_cam, perturbation_type="mask_shift_px_x", perturbation_value=shift))
            shifted = apply_mask_shift(base_mask, shift_x_px=0, shift_y_px=int(shift))
            rows.append(evaluate_arrays(metadata, gt_depth, gt_depth, shifted, intrinsics=intr, T_world_cam=T_world_cam, perturbation_type="mask_shift_px_y", perturbation_value=shift))

        for mode in mask_morphology_list:
            for kernel in mask_kernel_size_list:
                morphed = apply_mask_morphology(base_mask, mode=mode, kernel_size=int(kernel))
                rows.append(evaluate_arrays(metadata, gt_depth, gt_depth, morphed, intrinsics=intr, T_world_cam=T_world_cam, perturbation_type=f"mask_morphology_{mode}", perturbation_value=kernel))

        for trans_std in cam_trans_noise_std_list:
            T_pert = perturb_transform(T_world_cam, trans_noise_std_m=float(trans_std), rot_noise_deg=0.0, seed=_stable_seed(seed, metadata["sample_id"], f"trans_{trans_std}"))
            rows.append(evaluate_arrays(metadata, gt_depth, gt_depth, base_mask, intrinsics=intr, T_world_cam=T_pert, perturbation_type="cam_trans_noise_std_m", perturbation_value=trans_std))

        for rot_deg in cam_rot_noise_deg_list:
            T_pert = perturb_transform(T_world_cam, trans_noise_std_m=0.0, rot_noise_deg=float(rot_deg), seed=_stable_seed(seed, metadata["sample_id"], f"rot_{rot_deg}"))
            rows.append(evaluate_arrays(metadata, gt_depth, gt_depth, base_mask, intrinsics=intr, T_world_cam=T_pert, perturbation_type="cam_rot_noise_deg", perturbation_value=rot_deg))

        for fx_err in fx_relative_error_list:
            intr_pert = perturb_intrinsics(intr, fx_relative_error=float(fx_err))
            rows.append(evaluate_arrays(metadata, gt_depth, gt_depth, base_mask, intrinsics=intr_pert, T_world_cam=T_world_cam, perturbation_type="fx_relative_error", perturbation_value=fx_err))

        for fy_err in fy_relative_error_list:
            intr_pert = perturb_intrinsics(intr, fy_relative_error=float(fy_err))
            rows.append(evaluate_arrays(metadata, gt_depth, gt_depth, base_mask, intrinsics=intr_pert, T_world_cam=T_world_cam, perturbation_type="fy_relative_error", perturbation_value=fy_err))

        for cx_shift in cx_shift_px_list:
            intr_pert = perturb_intrinsics(intr, cx_shift_px=float(cx_shift))
            rows.append(evaluate_arrays(metadata, gt_depth, gt_depth, base_mask, intrinsics=intr_pert, T_world_cam=T_world_cam, perturbation_type="cx_shift_px", perturbation_value=cx_shift))

        for cy_shift in cy_shift_px_list:
            intr_pert = perturb_intrinsics(intr, cy_shift_px=float(cy_shift))
            rows.append(evaluate_arrays(metadata, gt_depth, gt_depth, base_mask, intrinsics=intr_pert, T_world_cam=T_world_cam, perturbation_type="cy_shift_px", perturbation_value=cy_shift))

        # Baseline error manifests as depth scale error after disparity-to-depth.
        baseline = float(metadata.get("baseline_m", 0.10))
        for rel_err in baseline_relative_error_list:
            scale = perturb_baseline(baseline, rel_err) / baseline
            pred = gt_depth * np.float32(scale)
            rows.append(evaluate_arrays(metadata, pred, gt_depth, base_mask, intrinsics=intr, T_world_cam=T_world_cam, perturbation_type="baseline_relative_error", perturbation_value=rel_err))

    report_dir = ensure_dir(reports_dir)
    write_csv(report_dir / "error_sensitivity_report.csv", rows, fieldnames=REPORT_FIELDS)
    write_json(report_dir / "error_sensitivity_report.json", rows)
    viz_dir = ensure_dir(visualizations_dir)
    _plot_series(rows, "depth_", viz_dir / "error_sensitivity_depth_noise.png", "Depth perturbation sensitivity")
    _plot_series(rows, "mask_shift", viz_dir / "error_sensitivity_mask_shift.png", "Mask shift sensitivity")
    _plot_series(rows, "cam_", viz_dir / "error_sensitivity_extrinsic_noise.png", "Camera extrinsic sensitivity")
    _plot_series(rows, "fx_", viz_dir / "error_sensitivity_intrinsics_baseline.png", "Intrinsics/baseline sensitivity")
    return rows

