# Stereo Spatial Calibration

Standalone IsaacLab project for validating a fixed stereo RGB spatial localization chain:

```text
left/right RGB
  -> FoundationStereo disparity
  -> metric depth
  -> camera-frame point cloud
  -> object mask point cloud
  -> object center
  -> world frame
  -> Franka base frame
  -> error report against GT pose
```

Current first version does not control the robot. It only validates visual spatial localization. FoundationStereo output is camera-frame geometry, not a robot executable target. Use `T_world_left_cam` and `T_base_world` from metadata to convert results into Franka base frame.

## Why Fixed Stereo First

Fixed stereo removes robot motion, hand-eye calibration changes, and timing coupling from the first validation pass. That makes the first acceptance target measurable: RGB stereo to depth/point cloud to object center to Franka base coordinates.

## Implemented Scope

- Resolution: `640 x 480`
- Baseline: `0.10 m`
- Camera frame convention: OpenCV, `x right`, `y down`, `z forward`
- Outputs: camera/world/base coordinates
- Objects: `cube`, `sphere`, `cylinder`
- Masks: Isaac GT mask contract, with synthetic fallback masks in this environment
- Center methods: `pointcloud_median`, `pointcloud_mean_filtered`
- Fallback: `--use_gt_depth_as_prediction true` copies GT depth into prediction for geometry-chain testing only

The Isaac native capture hook is isolated in `stereo_spatial_calib/isaac_scene_builder.py` and `stereo_spatial_calib/isaac_data_collector.py`. The stable acceptance path remains the deterministic synthetic rectified stereo backend, which writes the same file contract and keeps the fallback geometry chain testable without Isaac camera capture.

## Dependencies

Minimum fallback dependencies:

```bash
python -m pip install -U numpy pytest pillow pyyaml
```

Optional dependencies:

```bash
python -m pip install scipy matplotlib opencv-python open3d
```

FoundationStereo real inference additionally needs PyTorch/CUDA and a compatible FoundationStereo checkpoint.

## FoundationStereo

The environment checker searches:

- `~/FoundationStereo`
- `~/IsaacLab/FoundationStereo`
- `~/IsaacLab/third_party/FoundationStereo`

If none exists, it attempts:

```bash
git clone https://github.com/NVlabs/FoundationStereo.git ~/IsaacLab/third_party/FoundationStereo
```

Checkpoint discovery scans the repo for `*.pth`, `*.pt`, and `*.ckpt`. If no checkpoint is found, use fallback mode:

```bash
--use_gt_depth_as_prediction true
```

This fallback is only a geometry-chain test. It does not represent real visual inference.

## Commands

From `~/IsaacLab`:

```bash
source /home1/banghai/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/00_check_environment.py
```

Generate dataset:

```bash
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/01_generate_isaac_stereo_dataset.py \
  --num_samples_per_object 3 \
  --objects cube sphere cylinder \
  --resolution 640 480 \
  --baseline 0.10
```

Run FoundationStereo or fallback:

```bash
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/02_run_foundation_stereo.py \
  --dataset_dir source/standalone/stereo_spatial_calibration/outputs/dataset \
  --predictions_dir source/standalone/stereo_spatial_calibration/outputs/predictions \
  --use_gt_depth_as_prediction true
```

Reconstruct point clouds:

```bash
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/03_reconstruct_pointcloud.py
```

Evaluate object centers:

```bash
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/04_evaluate_spatial_localization.py
```

Full fallback pipeline:

```bash
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/run_full_pipeline.py \
  --num_samples_per_object 3 \
  --objects cube sphere cylinder \
  --resolution 640 480 \
  --baseline 0.10 \
  --backend synthetic \
  --depth_source gt
```

Full real FoundationStereo pipeline, after installing PyTorch/CUDA and a checkpoint:

```bash
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/run_full_pipeline.py \
  --num_samples_per_object 3 \
  --objects cube sphere cylinder \
  --resolution 640 480 \
  --baseline 0.10 \
  --backend synthetic \
  --depth_source foundation_stereo
```

`run_full_pipeline.py` cleans this project's managed `outputs` subdirectories by default. Pass `--clean_outputs false` to keep previous files.

## Tests

```bash
pytest source/standalone/stereo_spatial_calibration/tests -v
```

Tests cover disparity-to-depth, pixel backprojection, depth-to-pointcloud shape, mask extraction, transform/inverse/compose, and a no-Isaac smoke test.

## Output Files

- `outputs/dataset/{sample_id}/left_rgb.png`
- `outputs/dataset/{sample_id}/right_rgb.png`
- `outputs/dataset/{sample_id}/gt_depth_left.npy`
- `outputs/dataset/{sample_id}/gt_segmentation.npy`
- `outputs/dataset/{sample_id}/gt_instance_mask.npy`
- `outputs/dataset/{sample_id}/metadata.json`
- `outputs/predictions/{sample_id}/disparity.npy`
- `outputs/predictions/{sample_id}/pred_depth.npy`
- `outputs/predictions/{sample_id}/disparity_color.png`
- `outputs/predictions/{sample_id}/depth_color.png`
- `outputs/pointclouds/*_pred_camera.ply`
- `outputs/pointclouds/*_pred_world.ply`
- `outputs/pointclouds/*_object_pred_world.ply`
- `outputs/pointclouds/*_gt_world.ply`
- `outputs/reports/environment_report.json`
- `outputs/reports/spatial_localization_report.csv`
- `outputs/reports/spatial_localization_report.json`
- `outputs/visualizations/*_left_rgb_with_mask.png`
- `outputs/visualizations/*_pred_depth_color.png`
- `outputs/visualizations/*_gt_depth_color.png`
- `outputs/visualizations/*_depth_error_color.png`
- `outputs/visualizations/*_object_center_overlay.png`

## Actual Validation Results On This Machine

Environment check:

- Python: `/home1/banghai/miniconda3/envs/env_isaaclab/bin/python`, Python `3.11.15`
- Conda env: `env_isaaclab`
- `torch`: `2.7.0+cu128`
- CUDA: available
- GPU: `Quadro RTX 8000`
- `isaacsim`, `isaaclab`, `omni`, `isaacsim.simulation_app`: importable
- `open3d`: not installed; ASCII PLY/NPY output still works
- FoundationStereo repo: cloned/found at `~/IsaacLab/third_party/FoundationStereo`
- FoundationStereo checkpoint: not found
- Real FoundationStereo inference: not run
- Synthetic fallback GT-depth mode: complete

Commands run:

```bash
source /home1/banghai/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/00_check_environment.py
pytest source/standalone/stereo_spatial_calibration/tests -v
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/run_full_pipeline.py \
  --num_samples_per_object 1 \
  --objects cube sphere cylinder \
  --resolution 640 480 \
  --baseline 0.10 \
  --backend synthetic \
  --depth_source gt
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/05_run_error_sensitivity.py \
  --backend synthetic \
  --objects cube sphere cylinder \
  --num_samples_per_object 1 \
  --depth_noise_std_list 0.000 0.002 0.005 0.010 \
  --depth_bias_list -0.010 0.000 0.010 \
  --mask_shift_px_list -5 0 5 \
  --cam_trans_noise_std_list 0.000 0.002 0.005 \
  --cam_rot_noise_deg_list 0.0 0.5 1.0 \
  --fx_relative_error_list 0.0 0.005 0.010 \
  --baseline_relative_error_list 0.0 0.005 0.010 \
  --seed 42
```

Test result:

```text
23 passed
```

Fallback smoke errors from the final three-object run:

```text
cube_000001:     error_l2_mm = 0.648
cylinder_000001: error_l2_mm = 0.268
sphere_000001:   error_l2_mm = 0.244
```

Reports:

- `outputs/reports/environment_report.json`
- `outputs/reports/spatial_localization_report.csv`
- `outputs/reports/spatial_localization_report.json`
- `outputs/reports/error_sensitivity_report.csv`
- `outputs/reports/error_sensitivity_report.json`

## Troubleshooting

- `tabs: terminal type 'dumb' cannot reset tabs`: set `TERM=xterm` before `./isaaclab.sh`.
- No FoundationStereo checkpoint: keep using `--use_gt_depth_as_prediction true` for geometry tests, then place the official checkpoint under the FoundationStereo repo.
- `open3d` missing: the pipeline still writes ASCII PLY and NPY files.
- Negative or invalid disparity: verify left/right order, rectification, baseline in meters, and that disparity is in pixels.
- Large coordinate error: verify `T_world_left_cam`, `T_base_world`, camera convention conversion, and mask erosion settings.

## Extension Points

- `open_vocab_segmentation_adapter.py`: reserved for GroundingDINO + SAM, SAM-only, YOLO-seg, or similar mask providers.
- `learned_residual_calibrator.py`: reserved for learned residual correction from visual 3D coordinates to robot executable target coordinates.
- `franka_execution_adapter.py`: reserved for sending `center_base_m` to IK or a cerebellum/state-machine executor.

For Franka execution, the next contract should consume `center_base_m` from the JSON/CSV report, apply task-specific grasp offsets and safety checks, then pass a target pose to IK or the downstream controller. This repository's first version intentionally stops before command execution.

## Stage 2 Update

Stage 2 keeps the Stage 1 fallback path intact and adds runtime diagnostics, depth-source selection, FoundationStereo checkpoint-aware inference, and an Isaac native backend hook.

New scripts:

- `scripts/check_isaac_runtime.py`
- `scripts/check_foundation_stereo_runtime.py`
- `scripts/test_foundation_stereo_single_pair.py`

New reports:

- `outputs/reports/stage2_environment_diagnosis.json`
- `outputs/reports/stage2_runtime_recommendation.md`
- `outputs/reports/stage2_foundation_stereo_report.csv`
- `outputs/reports/stage2_foundation_stereo_report.json`
- `outputs/reports/stage2_depth_source_gt_report.csv`
- `outputs/reports/stage2_depth_source_noisy_gt_report.csv`
- `outputs/reports/isaac_native_backend_error.log`

### Runtime Diagnosis

The current unactivated shell uses base conda:

```text
./isaaclab.sh -p -> /home1/banghai/miniconda3/bin/python
CONDA_DEFAULT_ENV=base
torch: not importable
isaacsim: not importable
isaaclab: not importable
```

The existing Isaac environment is usable:

```bash
source /home1/banghai/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/00_check_environment.py
```

Confirmed in `env_isaaclab`:

```text
python: /home1/banghai/miniconda3/envs/env_isaaclab/bin/python
torch: 2.7.0+cu128
CUDA: true
GPU: Quadro RTX 8000
isaacsim: importable
isaaclab: 0.54.2
omni / isaacsim.simulation_app: importable
```

Do not install torch into base for this project. Activate `env_isaaclab` before real Isaac or FoundationStereo work.

### FoundationStereo Checkpoint

The local FoundationStereo README says to place the entire model folder under:

```text
third_party/FoundationStereo/pretrained_models/23-51-11/
```

Expected files:

```text
third_party/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
third_party/FoundationStereo/pretrained_models/23-51-11/cfg.yaml
```

Current status:

```text
FoundationStereo repo: ~/IsaacLab/third_party/FoundationStereo
checkpoint: not found
real inference: not run
single-pair smoke: skipped cleanly
```

Once the checkpoint is present, run:

```bash
source /home1/banghai/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd ~/IsaacLab
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/run_full_pipeline.py \
  --num_samples_per_object 1 \
  --objects cube sphere cylinder \
  --resolution 640 480 \
  --baseline 0.10 \
  --backend synthetic \
  --depth_source foundation_stereo
```

### Depth Sources

`run_full_pipeline.py` now supports:

```text
--depth_source gt
--depth_source noisy_gt --depth_noise_std_m 0.005 --depth_noise_bias_m 0.0
--depth_source foundation_stereo
```

Actual Stage 2 comparison:

```text
gt:
  cube_000001     error_l2_mm=0.648, depth_rmse_m=0.000000
  cylinder_000001 error_l2_mm=0.268, depth_rmse_m=0.000000
  sphere_000001   error_l2_mm=0.244, depth_rmse_m=0.000000

noisy_gt, std=0.005 m:
  cube_000001     error_l2_mm=0.635, depth_rmse_m=0.004883
  cylinder_000001 error_l2_mm=0.120, depth_rmse_m=0.004900
  sphere_000001   error_l2_mm=0.284, depth_rmse_m=0.004968
```

The object-center L2 error is median-based and can move non-monotonically for a single sample, but the depth RMSE increases as expected and is covered by `tests/test_depth_noise.py`.

### Isaac Native Backend

The synthetic backend remains the default:

```bash
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/run_full_pipeline.py \
  --num_samples_per_object 1 \
  --objects cube sphere cylinder \
  --resolution 640 480 \
  --baseline 0.10 \
  --backend synthetic \
  --depth_source gt
```

The `isaac_native` hook now starts Isaac Sim under `env_isaaclab` and creates the native USD scene primitives. Full RGB/depth/instance camera capture is not finalized for this Isaac Sim runtime yet; the failure is written to:

```text
outputs/reports/isaac_native_backend_error.log
```

### Stage 2 Validation Commands Run

```bash
pytest source/standalone/stereo_spatial_calibration/tests -v
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/00_check_environment.py
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/run_full_pipeline.py \
  --num_samples_per_object 1 \
  --objects cube sphere cylinder \
  --resolution 640 480 \
  --baseline 0.10 \
  --backend synthetic \
  --depth_source gt
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/run_full_pipeline.py \
  --num_samples_per_object 1 \
  --objects cube sphere cylinder \
  --resolution 640 480 \
  --baseline 0.10 \
  --backend synthetic \
  --depth_source noisy_gt \
  --depth_noise_std_m 0.005
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/test_foundation_stereo_single_pair.py
```

Result:

```text
pytest: 23 passed
synthetic + gt: passed
synthetic + noisy_gt: passed
FoundationStereo single-pair: skipped because checkpoint is missing
isaac_native: SimulationApp/scene hook starts; current camera capture still fails before RGB/depth artifacts are produced
```

## Stage 3 Update

Stage 3 adds controlled perturbation tools and an error sensitivity runner while keeping the Stage 1/2 fallback path intact.

New code:

- `stereo_spatial_calib/error_sensitivity.py`
- `scripts/05_run_error_sensitivity.py`
- `scripts/test_isaac_native_camera_capture.py`
- perturbation helpers in `camera_geometry.py`, `segmentation_utils.py`, `coordinate_transform.py`, and `evaluator.py`

New tests:

- `tests/test_error_sensitivity.py`
- `tests/test_mask_perturbation.py`
- `tests/test_intrinsics_perturbation.py`
- `tests/test_transform_perturbation.py`

Run the native camera capture smoke test:

```bash
source /home1/banghai/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd ~/IsaacLab
TERM=xterm timeout 90s ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/test_isaac_native_camera_capture.py
```

Current native status:

```text
SimulationApp starts: yes
USD scene/camera hook: yes
RGB/depth capture: not complete in this runtime
latest failure record: outputs/reports/isaac_native_backend_error.log
report: outputs/reports/isaac_native_camera_capture_report.json
```

The current failure is in the native camera frame acquisition layer. The first Camera wrapper attempt timed out waiting for valid `get_rgb()` / `get_depth()` data. The next attempt exposed that Isaac Sim 5.1 in this environment does not expose `omni.syntheticdata.sensors` as used by some bundled tests. The collector now uses a synchronous Replicator-compatible step fallback; the remaining API point to verify is the exact headless render-product/annotator stepping method for this installed Isaac Sim build.

Run error sensitivity:

```bash
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/05_run_error_sensitivity.py \
  --backend synthetic \
  --objects cube sphere cylinder \
  --num_samples_per_object 1 \
  --depth_noise_std_list 0.000 0.002 0.005 0.010 \
  --depth_bias_list -0.010 0.000 0.010 \
  --mask_shift_px_list -5 0 5 \
  --cam_trans_noise_std_list 0.000 0.002 0.005 \
  --cam_rot_noise_deg_list 0.0 0.5 1.0 \
  --fx_relative_error_list 0.0 0.005 0.010 \
  --baseline_relative_error_list 0.0 0.005 0.010 \
  --seed 42
```

Sensitivity outputs:

- `outputs/reports/error_sensitivity_report.csv`
- `outputs/reports/error_sensitivity_report.json`
- `outputs/visualizations/error_sensitivity_depth_noise.png`
- `outputs/visualizations/error_sensitivity_mask_shift.png`
- `outputs/visualizations/error_sensitivity_extrinsic_noise.png`
- `outputs/visualizations/error_sensitivity_intrinsics_baseline.png`

Observed largest errors in the Stage 3 run:

```text
sphere_000001 cam_rot_noise_deg=1.0 -> error_l2_mm=13.454
cylinder_000001 cam_rot_noise_deg=1.0 -> error_l2_mm=13.318
cube_000001 cam_rot_noise_deg=1.0 -> error_l2_mm=11.449
depth_bias_m +/-0.010 m -> about 10 mm object-center error
```

Depth random noise may not strongly move object center estimates because the default median/filtered mean suppresses zero-mean noise inside the mask. Depth bias, mask shift/dilation, camera extrinsic rotation, intrinsics error, and baseline error are closer to real systematic calibration failures and are the useful signals for later residual correction and spatial calibration design.

## Stage 4 Update

Stage 4 adds a native capture method selector under the existing `--backend isaac_native` path:

- `--native_capture_method camera_class`: `isaacsim.sensors.camera.Camera`
- `--native_capture_method replicator_annotator`: `omni.replicator.core` render products and annotators
- `--native_capture_method tiled_camera`: import check for `isaaclab.sensors.TiledCamera`
- `--native_capture_method auto`: tries the three methods in that order

New/updated files:

- `stereo_spatial_calib/native_capture_methods.py`
- `stereo_spatial_calib/metadata_validator.py`
- `stereo_spatial_calib/isaac_scene_builder.py`
- `stereo_spatial_calib/isaac_data_collector.py`
- `scripts/test_isaac_native_camera_capture.py`
- `tests/test_metadata_validator.py`
- `tests/test_projected_mask_fallback.py`
- `tests/test_native_capture_result_schema.py`

Native capture test command:

```bash
source /home1/banghai/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd ~/IsaacLab
TERM=xterm timeout 160s ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/test_isaac_native_camera_capture.py \
  --native_capture_method auto \
  --resolution 640 480 \
  --baseline 0.10
```

Current Stage 4 result on this machine:

```text
SimulationApp startup: passed
Isaac Core World scene creation: passed
camera/render-product buffer allocation: passed
camera_class: failed with all-black RGB and invalid depth
replicator_annotator: failed with all-black RGB and invalid depth
tiled_camera: importable, not wired to an IsaacLab interactive scene yet
native full pipeline: not passed yet
synthetic full pipeline: passed
pytest: 28 passed
```

Current native failure signature:

```text
left_rgb_shape=[480, 640, 3]
right_rgb_shape=[480, 640, 3]
depth_shape=[480, 640]
left_rgb_mean=0.0
depth_valid_ratio=0.0
```

The failure is now isolated to Isaac Sim 5.1 headless render-product/timeline synchronization: native methods return correctly shaped buffers, but the buffers are all-black and depth has no finite positive values. The reports are:

- `outputs/reports/stage4_native_failure_diagnosis.md`
- `outputs/reports/isaac_native_backend_error.log`
- `outputs/reports/isaac_native_camera_capture_report.json`

The projected mask fallback remains implemented in `native_capture_methods.py` and records `mask_source="projected_bbox_fallback"` when segmentation is not available. It is a temporary geometry-based mask, not a replacement for true instance segmentation.

Stage 4 validation commands run:

```bash
pytest source/standalone/stereo_spatial_calibration/tests -v

TERM=xterm timeout 160s ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/test_isaac_native_camera_capture.py \
  --native_capture_method auto \
  --resolution 640 480 \
  --baseline 0.10

TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/run_full_pipeline.py \
  --num_samples_per_object 1 \
  --objects cube sphere cylinder \
  --resolution 640 480 \
  --baseline 0.10 \
  --backend synthetic \
  --depth_source gt
```

Latest synthetic result:

```text
cube_000001: error_l2_mm=0.648
cylinder_000001: error_l2_mm=0.268
sphere_000001: error_l2_mm=0.244
```

FoundationStereo remains cleanly skipped because no checkpoint is installed. Next checkpoint command once the model is placed under `third_party/FoundationStereo/pretrained_models/23-51-11/`:

```bash
TERM=xterm ./isaaclab.sh -p source/standalone/stereo_spatial_calibration/scripts/run_full_pipeline.py \
  --num_samples_per_object 1 \
  --objects cube \
  --resolution 640 480 \
  --baseline 0.10 \
  --backend synthetic \
  --depth_source foundation_stereo
```

Next integration points:

- SAM / YOLO-seg / GroundingDINO: `stereo_spatial_calib/open_vocab_segmentation_adapter.py`
- residual correction and calibration: `stereo_spatial_calib/learned_residual_calibrator.py`
- Franka execution bridge: `stereo_spatial_calib/franka_execution_adapter.py`
