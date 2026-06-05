# Stage 4 Isaac Native Failure Diagnosis

Date: 2026-06-05

## Summary

The current failure is in native render-product frame acquisition, not in Python environment setup, SimulationApp startup, scene creation, or camera prim construction.

Observed status from the Stage 4 native runs:

- `env_isaaclab` is active and correct.
- SimulationApp starts successfully.
- Isaac Core `World` scene creation path was added and executed.
- Native camera/render-product buffers are allocated with expected image shapes.
- `camera_class` produced arrays shaped `left_rgb=[480, 640, 3]`, `right_rgb=[480, 640, 3]`, `depth=[480, 640]`, but RGB was all black and depth had `depth_valid_ratio=0.0`.
- `replicator_annotator` produced the same all-black RGB and invalid depth.
- `tiled_camera` is importable but not wired to an IsaacLab interactive scene yet.

## Failure Location

- SimulationApp startup: passed
- scene creation: passed for USD path and Isaac Core `World` path
- camera prim creation: passed
- render product creation: passed enough to return shaped buffers
- RGB acquisition: failed semantically, because returned RGB is all zeros
- depth acquisition: failed semantically, because returned depth has no finite positive values
- segmentation acquisition: not reached; projected bbox fallback remains available
- timeline/update/step synchronization: still suspected root cause
- data conversion/save: not reached for native, because acquisition is rejected before saving

## Methods Tried

1. `camera_class`
   - Uses `isaacsim.sensors.camera.Camera`.
   - Adds `rgb`, `distance_to_image_plane`, `distance_to_camera`, and pointcloud fallback.
   - Uses `omni.syntheticdata.sensors.next_render_simulation_async` through Kit async engine.
   - Current result: shaped buffers, all-black RGB, invalid depth.

2. `replicator_annotator`
   - Uses `omni.replicator.core` render products and `rgb`, `distance_to_image_plane`, `distance_to_camera`, segmentation annotators.
   - Current result: shaped buffers, all-black RGB, invalid depth.

3. `tiled_camera`
   - `isaaclab.sensors.TiledCamera` imports.
   - Current result: explicitly not wired to an IsaacLab interactive/manager scene.

## Current Native Error Signature

```text
camera_class did not produce valid RGB/depth frames:
left_rgb_shape=[480, 640, 3]
right_rgb_shape=[480, 640, 3]
depth_shape=[480, 640]
depth_valid_ratio=0.0
left_rgb_mean=0.0
left_rgb_min=0
left_rgb_max=0

replicator_annotator did not produce valid RGB/depth frames:
left_rgb_shape=[480, 640, 3]
right_rgb_shape=[480, 640, 3]
depth_shape=[480, 640]
depth_valid_ratio=0.0
left_rgb_mean=0.0
left_rgb_min=0
left_rgb_max=0
```

## Current Conclusion

Stage 4 native acquisition is not yet fully passed on this machine. The code now exposes three capture methods and records the right failure boundary. The next fix should focus on Isaac Sim 5.1 headless render-product/timeline synchronization, likely by reproducing a minimal built-in camera test outside this project and then porting the exact update/orchestrator sequence back into `native_capture_methods.py`.

Synthetic backend remains the validated path and still passes the full localization pipeline.
