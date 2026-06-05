# Stereo Spatial Calibration Interactive Demo Debug Package

## Purpose
Package the interactive Isaac Sim stereo capture demo without checkpoints or large data.

## Source
`~/IsaacLab/source/standalone/stereo_spatial_calibration`

## Current State
- pytest: 45 passed
- synthetic backend: cube 0.648 mm, cylinder 0.268 mm, sphere 0.244 mm
- native RGB/depth and native instance segmentation: working
- FoundationStereo true inference: working
- interactive headless smoke: success
- interactive depth_source: foundation_stereo_depth
- interactive mask_source: native_instance
- pointcloud_reference_view: left
- interactive runtime_ms: 25110.678434022702
- interactive error_l2_mm: 64.34965133666992
- report_html_path: /home1/banghai/IsaacLab/source/standalone/stereo_spatial_calibration/outputs/interactive_demo/latest/report.html

## Run Commands
```bash
source /home1/banghai/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd ~/IsaacLab
pytest source/standalone/stereo_spatial_calibration/tests -v
ENABLE_CAMERAS=1 TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/launch_interactive_demo.py --headless_smoke_test --baseline 0.10 --resolution 640 480
ENABLE_CAMERAS=1 TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/launch_interactive_demo.py --gui --baseline 0.10 --resolution 640 480
```

## Exclusions
- FoundationStereo checkpoints and all `*.pth`/`*.pt`/`*.ckpt` files excluded.
- Large generated datasets, predictions, conda envs, and Isaac Sim installation files excluded.
- `*.npy`/`*.npz` arrays excluded.

## Large Files Over 20MB
- None

## Copied Files
- `README.md`
- `configs/camera_stereo.yaml`
- `configs/default.yaml`
- `exts/stereo_spatial_demo/config/extension.toml`
- `exts/stereo_spatial_demo/stereo_spatial_demo/__init__.py`
- `exts/stereo_spatial_demo/stereo_spatial_demo/async_inference_worker.py`
- `exts/stereo_spatial_demo/stereo_spatial_demo/demo_controller.py`
- `exts/stereo_spatial_demo/stereo_spatial_demo/extension.py`
- `exts/stereo_spatial_demo/stereo_spatial_demo/ui_builder.py`
- `outputs/interactive_demo/latest/depth_error_color.png`
- `outputs/interactive_demo/latest/disparity_color.png`
- `outputs/interactive_demo/latest/gt_depth_color.png`
- `outputs/interactive_demo/latest/left_rgb.png`
- `outputs/interactive_demo/latest/mask_overlay.png`
- `outputs/interactive_demo/latest/object_center_overlay.png`
- `outputs/interactive_demo/latest/object_pointcloud_world.ply`
- `outputs/interactive_demo/latest/overview.png`
- `outputs/interactive_demo/latest/pred_depth_color.png`
- `outputs/interactive_demo/latest/report.html`
- `outputs/interactive_demo/latest/report.json`
- `outputs/interactive_demo/latest/report.md`
- `outputs/interactive_demo/latest/right_rgb.png`
- `outputs/reports/environment_report.json`
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
- `scripts/launch_interactive_demo.py`
- `scripts/run_full_pipeline.py`
- `scripts/test_foundation_stereo_native_pair.py`
- `scripts/test_foundation_stereo_single_pair.py`
- `scripts/test_isaac_native_camera_capture.py`
- `stereo_spatial_calib/__init__.py`
- `stereo_spatial_calib/async_inference_worker.py`
- `stereo_spatial_calib/camera_geometry.py`
- `stereo_spatial_calib/coordinate_transform.py`
- `stereo_spatial_calib/error_sensitivity.py`
- `stereo_spatial_calib/evaluator.py`
- `stereo_spatial_calib/foundation_stereo_adapter.py`
- `stereo_spatial_calib/foundation_stereo_checkpoint.py`
- `stereo_spatial_calib/franka_execution_adapter.py`
- `stereo_spatial_calib/interactive_demo_runner.py`
- `stereo_spatial_calib/interactive_visualization.py`
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
- `stereo_spatial_calib/stereo_rig_controller.py`
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
- `tests/test_interactive_demo_runner_schema.py`
- `tests/test_intrinsics_perturbation.py`
- `tests/test_mask_perturbation.py`
- `tests/test_mask_source_selection.py`
- `tests/test_metadata_validator.py`
- `tests/test_native_capture_result_schema.py`
- `tests/test_native_segmentation_schema.py`
- `tests/test_pipeline_smoke.py`
- `tests/test_pointcloud_reference_view.py`
- `tests/test_pointcloud_utils.py`
- `tests/test_projected_mask_fallback.py`
- `tests/test_stereo_rig_controller.py`
- `tests/test_stereo_rig_validation.py`
- `tests/test_transform_perturbation.py`
- `tests/test_visual_report_generation.py`
