# Stereo Spatial Calibration Stage 4 Debug Package

## Purpose

This package is a small GitHub-friendly snapshot for analyzing why Isaac Sim native RGB/depth acquisition returns black RGB frames and invalid depth in the Stage 4 stereo spatial calibration work.

It intentionally excludes model weights, checkpoints, Isaac Sim installations, conda environments, large datasets, predictions, and point clouds.

## Source Project

`~/IsaacLab/source/standalone/stereo_spatial_calibration`

## Known Current Status

- pytest: 28 passed
- synthetic backend: passed
- native SimulationApp startup: passed
- native scene creation: passed
- native camera/render product buffer allocation: passed, returning `480x640` shaped buffers
- RGB acquisition: returns black frames, `left_rgb_mean=0.0`
- depth acquisition: invalid, `depth_valid_ratio=0.0`
- FoundationStereo repo exists at `~/IsaacLab/third_party/FoundationStereo`
- FoundationStereo checkpoint is missing
- real FoundationStereo inference cleanly skips

## Current Analysis Goal

Please analyze:

- `stereo_spatial_calib/native_capture_methods.py`
- `stereo_spatial_calib/isaac_scene_builder.py`
- `stereo_spatial_calib/isaac_data_collector.py`
- `outputs/reports/stage4_native_failure_diagnosis.md`
- `outputs/reports/isaac_native_backend_error.log`
- `outputs/reports/isaac_native_camera_capture_report.json`

The goal is to locate why Isaac native camera capture produces all-black RGB and no valid depth in this Isaac Sim 5.1 headless runtime.

## Key Commands

```bash
source /home1/banghai/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd ~/IsaacLab

pytest source/standalone/stereo_spatial_calibration/tests -v

./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/test_isaac_native_camera_capture.py \
  --native_capture_method auto \
  --resolution 640 480 \
  --baseline 0.10

./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/run_full_pipeline.py \
  --num_samples_per_object 1 \
  --objects cube sphere cylinder \
  --resolution 640 480 \
  --baseline 0.10 \
  --backend synthetic \
  --depth_source gt
```

## Copied Files

```text
.gitignore
MANIFEST.md
README.md
configs/camera_stereo.yaml
configs/default.yaml
outputs/reports/isaac_native_backend_error.log
outputs/reports/isaac_native_camera_capture_report.json
outputs/reports/spatial_localization_report.csv
outputs/reports/spatial_localization_report.json
outputs/reports/stage2_environment_diagnosis.json
outputs/reports/stage2_runtime_recommendation.md
outputs/reports/stage4_native_failure_diagnosis.md
scripts/00_check_environment.py
scripts/01_generate_isaac_stereo_dataset.py
scripts/02_run_foundation_stereo.py
scripts/03_reconstruct_pointcloud.py
scripts/04_evaluate_spatial_localization.py
scripts/05_run_error_sensitivity.py
scripts/check_foundation_stereo_runtime.py
scripts/check_isaac_runtime.py
scripts/run_full_pipeline.py
scripts/test_foundation_stereo_single_pair.py
scripts/test_isaac_native_camera_capture.py
stereo_spatial_calib/__init__.py
stereo_spatial_calib/camera_geometry.py
stereo_spatial_calib/coordinate_transform.py
stereo_spatial_calib/error_sensitivity.py
stereo_spatial_calib/evaluator.py
stereo_spatial_calib/foundation_stereo_adapter.py
stereo_spatial_calib/franka_execution_adapter.py
stereo_spatial_calib/io_utils.py
stereo_spatial_calib/isaac_data_collector.py
stereo_spatial_calib/isaac_scene_builder.py
stereo_spatial_calib/learned_residual_calibrator.py
stereo_spatial_calib/metadata_validator.py
stereo_spatial_calib/native_capture_methods.py
stereo_spatial_calib/open_vocab_segmentation_adapter.py
stereo_spatial_calib/pointcloud_utils.py
stereo_spatial_calib/runtime_diagnostics.py
stereo_spatial_calib/segmentation_utils.py
stereo_spatial_calib/visualization.py
tests/conftest.py
tests/test_backend_selection.py
tests/test_camera_geometry.py
tests/test_coordinate_transform.py
tests/test_depth_noise.py
tests/test_error_sensitivity.py
tests/test_foundation_stereo_adapter_skip.py
tests/test_intrinsics_perturbation.py
tests/test_mask_perturbation.py
tests/test_metadata_validator.py
tests/test_native_capture_result_schema.py
tests/test_pipeline_smoke.py
tests/test_pointcloud_utils.py
tests/test_projected_mask_fallback.py
tests/test_transform_perturbation.py
```

## Missing Files

These requested files were not present in the current source outputs at packaging time:

```text
outputs/reports/error_sensitivity_report.csv
outputs/reports/error_sensitivity_report.json
outputs/dataset/isaac_native_camera_test/left_rgb.png
outputs/dataset/isaac_native_camera_test/right_rgb.png
outputs/dataset/isaac_native_camera_test/metadata.json
```

## Excluded Large Or Sensitive Files

No files larger than 20 MB were found in the debug package after copying.

No model/checkpoint files were found in the debug package after copying:

```text
*.pth
*.pt
*.onnx
*.engine
*.ckpt
```

The following output classes were deliberately excluded:

```text
outputs/dataset/**
outputs/pointclouds/**
outputs/predictions/**
third_party/
*.npy
*.npz
```
