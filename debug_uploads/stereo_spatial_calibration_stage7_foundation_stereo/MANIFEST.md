# Stereo Spatial Calibration Stage 7 FoundationStereo Debug Package

## Purpose
Package the Isaac native stereo + native mask + true FoundationStereo integration for review without uploading checkpoints or large generated data.

## Source Project
`~/IsaacLab/source/standalone/stereo_spatial_calibration`

## Current State
- pytest: 39 passed
- synthetic backend: passes with cube 0.648 mm, cylinder 0.268 mm, sphere 0.244 mm
- native RGB/depth acquisition: passes
- native instance segmentation: passes
- native + GT depth + native instance mask: cube error_l2_mm=8.928
- FoundationStereo repo: `~/IsaacLab/third_party/FoundationStereo`
- FoundationStereo checkpoint: present locally but excluded from this package
- FoundationStereo native pair test: success
- native pair runtime_ms: 24641.77829993423
- native pair pred_depth_valid_ratio: 1.0
- native pair depth_rmse_m: 0.008722597733139992
- native stereo RGB -> FoundationStereo -> native mask -> coordinate: success
- stage7 error_l2_mm: 9.252578020095825
- center_pred_world_m: [0.5364290475845337, 0.20272007584571838, 0.04821479320526123]
- center_gt_world_m: [0.5375286340713501, 0.19860689342021942, 0.03999999910593033]
- depth_rmse_m: 0.0030131845269352198
- pred_depth_valid_ratio: 1.0

## Key Commands
```bash
source /home1/banghai/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd ~/IsaacLab
pytest source/standalone/stereo_spatial_calibration/tests -v
ENABLE_CAMERAS=1 TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/test_foundation_stereo_native_pair.py --resolution 640 480 --baseline 0.10 --mask_source auto
ENABLE_CAMERAS=1 TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/run_full_pipeline.py --num_samples_per_object 1 --objects cube --resolution 640 480 --baseline 0.10 --backend isaac_native --native_capture_method camera_class --depth_source foundation_stereo --mask_source auto
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/07_generate_visual_debug_report.py
```

## Exclusions
- FoundationStereo checkpoint files (`*.pth`, `*.pt`, `*.ckpt`) are excluded.
- `outputs/predictions`, `outputs/pointclouds`, large datasets, conda envs, and Isaac Sim installation files are excluded.
- `*.npy`/`*.npz` generated arrays are excluded from the debug upload package.

## Large Files Over 20MB
- None

## Copied Files

- `README.md`
- `configs/camera_stereo.yaml`
- `configs/default.yaml`
- `outputs/reports/environment_report.json`
- `outputs/reports/foundation_stereo_checkpoint_download_report.json`
- `outputs/reports/foundation_stereo_checkpoint_layout.md`
- `outputs/reports/foundation_stereo_native_pair_report.json`
- `outputs/reports/isaac_native_camera_capture_report.json`
- `outputs/reports/spatial_localization_report.csv`
- `outputs/reports/spatial_localization_report.json`
- `outputs/reports/stage2_environment_diagnosis.json`
- `outputs/reports/stage2_foundation_stereo_report.csv`
- `outputs/reports/stage2_foundation_stereo_report.json`
- `outputs/reports/stage2_runtime_recommendation.md`
- `outputs/reports/stage4_isaac_native_report.csv`
- `outputs/reports/stage4_isaac_native_report.json`
- `outputs/reports/stage6_native_segmentation_report.csv`
- `outputs/reports/stage6_native_segmentation_report.json`
- `outputs/reports/stage7_foundation_stereo_native_report.csv`
- `outputs/reports/stage7_foundation_stereo_native_report.json`
- `outputs/reports/visual_debug_report.md`
- `outputs/reports/visual_debug_report_summary.json`
- `outputs/visualizations/cube_000001_depth_error_color.png`
- `outputs/visualizations/cube_000001_disparity_color.png`
- `outputs/visualizations/cube_000001_gt_depth_color.png`
- `outputs/visualizations/cube_000001_left_rgb_with_mask.png`
- `outputs/visualizations/cube_000001_object_center_overlay.png`
- `outputs/visualizations/cube_000001_pred_depth_color.png`
- `outputs/visualizations/foundation_stereo_native_pair_depth_error.png`
- `outputs/visualizations/foundation_stereo_native_pair_disparity.png`
- `outputs/visualizations/foundation_stereo_native_pair_gt_depth.png`
- `outputs/visualizations/foundation_stereo_native_pair_left_rgb.png`
- `outputs/visualizations/foundation_stereo_native_pair_mask_overlay.png`
- `outputs/visualizations/foundation_stereo_native_pair_pred_depth.png`
- `outputs/visualizations/foundation_stereo_native_pair_right_rgb.png`
- `scripts/00_check_environment.py`
- `scripts/01_generate_isaac_stereo_dataset.py`
- `scripts/02_run_foundation_stereo.py`
- `scripts/03_reconstruct_pointcloud.py`
- `scripts/04_evaluate_spatial_localization.py`
- `scripts/05_run_error_sensitivity.py`
- `scripts/06_compare_mask_sources.py`
- `scripts/07_generate_visual_debug_report.py`
- `scripts/check_foundation_stereo_runtime.py`
- `scripts/check_isaac_runtime.py`
- `scripts/debug_minimal_isaac_camera_offscreen.py`
- `scripts/debug_native_segmentation.py`
- `scripts/download_foundation_stereo_checkpoint.py`
- `scripts/run_full_pipeline.py`
- `scripts/test_foundation_stereo_native_pair.py`
- `scripts/test_foundation_stereo_single_pair.py`
- `scripts/test_isaac_native_camera_capture.py`
- `stereo_spatial_calib/__init__.py`
- `stereo_spatial_calib/camera_geometry.py`
- `stereo_spatial_calib/coordinate_transform.py`
- `stereo_spatial_calib/error_sensitivity.py`
- `stereo_spatial_calib/evaluator.py`
- `stereo_spatial_calib/foundation_stereo_adapter.py`
- `stereo_spatial_calib/foundation_stereo_checkpoint.py`
- `stereo_spatial_calib/franka_execution_adapter.py`
- `stereo_spatial_calib/io_utils.py`
- `stereo_spatial_calib/isaac_data_collector.py`
- `stereo_spatial_calib/isaac_scene_builder.py`
- `stereo_spatial_calib/learned_residual_calibrator.py`
- `stereo_spatial_calib/metadata_validator.py`
- `stereo_spatial_calib/native_capture_methods.py`
- `stereo_spatial_calib/native_segmentation.py`
- `stereo_spatial_calib/open_vocab_segmentation_adapter.py`
- `stereo_spatial_calib/pointcloud_utils.py`
- `stereo_spatial_calib/runtime_diagnostics.py`
- `stereo_spatial_calib/segmentation_utils.py`
- `stereo_spatial_calib/visualization.py`
- `tests/conftest.py`
- `tests/test_backend_selection.py`
- `tests/test_camera_geometry.py`
- `tests/test_coordinate_transform.py`
- `tests/test_depth_noise.py`
- `tests/test_depth_source_foundation_stereo.py`
- `tests/test_error_sensitivity.py`
- `tests/test_foundation_stereo_adapter_skip.py`
- `tests/test_foundation_stereo_checkpoint_detection.py`
- `tests/test_foundation_stereo_download_script.py`
- `tests/test_intrinsics_perturbation.py`
- `tests/test_mask_perturbation.py`
- `tests/test_mask_source_selection.py`
- `tests/test_metadata_validator.py`
- `tests/test_native_capture_result_schema.py`
- `tests/test_native_segmentation_schema.py`
- `tests/test_pipeline_smoke.py`
- `tests/test_pointcloud_utils.py`
- `tests/test_projected_mask_fallback.py`
- `tests/test_transform_perturbation.py`
- `tests/test_visual_report_generation.py`
