# Stereo Spatial Calibration Visual Debug Report

- sample_id: `cube_000001`
- backend: `isaac_native`
- native_capture_method: `camera_class`
- mask_source: `native_instance`
- segmentation_available: `True`
- depth_source: `gt_depth`
- FoundationStereo checkpoint_ready: `False`
- FoundationStereo inference_success: `False`
- FoundationStereo runtime_status: `success`

## Pipeline Status

- native RGB/depth available: `True`
- segmentation available: `True`
- checkpoint missing: `False`
- showing FoundationStereo depth: `False`

If `checkpoint_missing=true`, FoundationStereo images are intentionally omitted. If `depth_source=gt_depth`, the depth images are GT-depth geometry validation outputs, not FoundationStereo predictions.

## Coordinates

- center_camera_m: `[0.03643251582980156, -0.053373485803604126, 2.2454991340637207]`
- center_world_m: `[0.5364325046539307, 0.20032638311386108, 0.04869195818901062]`
- center_base_m: `[0.5364325046539307, 0.20032638311386108, 0.04869195818901062]`
- center_gt_world_m: `[0.5375286340713501, 0.19860689342021942, 0.03999999910593033]`
- error_l2_mm: `8.927950635552406`

## Images

![left_rgb.png](/home1/banghai/IsaacLab/source/standalone/stereo_spatial_calibration/outputs/dataset/cube_000001/left_rgb.png)

![right_rgb.png](/home1/banghai/IsaacLab/source/standalone/stereo_spatial_calibration/outputs/dataset/cube_000001/right_rgb.png)

![cube_000001_left_rgb_with_mask.png](/home1/banghai/IsaacLab/source/standalone/stereo_spatial_calibration/outputs/visualizations/cube_000001_left_rgb_with_mask.png)

![cube_000001_gt_depth_color.png](/home1/banghai/IsaacLab/source/standalone/stereo_spatial_calibration/outputs/visualizations/cube_000001_gt_depth_color.png)

![cube_000001_object_center_overlay.png](/home1/banghai/IsaacLab/source/standalone/stereo_spatial_calibration/outputs/visualizations/cube_000001_object_center_overlay.png)

## Depth Metrics

- depth_mae_m: `0.0`
- depth_rmse_m: `0.0`
- depth_error_median_m: `0.0`
- depth_error_p90_m: `0.0`
- depth_error_p95_m: `0.0`
- pred_depth_valid_ratio: `1.0`

## Point Clouds

- pointcloud_dir: `/home1/banghai/IsaacLab/source/standalone/stereo_spatial_calibration/outputs/pointclouds`
