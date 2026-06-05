# Stereo Spatial Calibration Visual Debug Report

- sample_id: `cube_000001`
- backend: `isaac_native`
- native_capture_method: `camera_class`
- mask_source: `native_instance`
- segmentation_available: `True`
- depth_source: `foundation_stereo_depth`
- FoundationStereo checkpoint_ready: `True`
- FoundationStereo inference_success: `True`
- FoundationStereo runtime_status: `success`

## Pipeline Status

- native RGB/depth available: `True`
- segmentation available: `True`
- checkpoint missing: `False`
- showing FoundationStereo depth: `True`

If `checkpoint_missing=true`, FoundationStereo images are intentionally omitted. If `depth_source=gt_depth`, the depth images are GT-depth geometry validation outputs, not FoundationStereo predictions.

## Coordinates

- center_camera_m: `[0.036429036408662796, -0.0534450002014637, 2.247938871383667]`
- center_world_m: `[0.5364290475845337, 0.20272007584571838, 0.04821479320526123]`
- center_base_m: `[0.5364290475845337, 0.20272007584571838, 0.04821479320526123]`
- center_gt_world_m: `[0.5375286340713501, 0.19860689342021942, 0.03999999910593033]`
- error_l2_mm: `9.252578020095825`

## Images

![left_rgb.png](/home1/banghai/IsaacLab/source/standalone/stereo_spatial_calibration/outputs/dataset/cube_000001/left_rgb.png)

![right_rgb.png](/home1/banghai/IsaacLab/source/standalone/stereo_spatial_calibration/outputs/dataset/cube_000001/right_rgb.png)

![cube_000001_left_rgb_with_mask.png](/home1/banghai/IsaacLab/source/standalone/stereo_spatial_calibration/outputs/visualizations/cube_000001_left_rgb_with_mask.png)

![cube_000001_gt_depth_color.png](/home1/banghai/IsaacLab/source/standalone/stereo_spatial_calibration/outputs/visualizations/cube_000001_gt_depth_color.png)

![disparity_color.png](/home1/banghai/IsaacLab/source/standalone/stereo_spatial_calibration/outputs/predictions/cube_000001/disparity_color.png)

![pred_depth_color.png](/home1/banghai/IsaacLab/source/standalone/stereo_spatial_calibration/outputs/predictions/cube_000001/pred_depth_color.png)

![cube_000001_depth_error_color.png](/home1/banghai/IsaacLab/source/standalone/stereo_spatial_calibration/outputs/visualizations/cube_000001_depth_error_color.png)

![cube_000001_object_center_overlay.png](/home1/banghai/IsaacLab/source/standalone/stereo_spatial_calibration/outputs/visualizations/cube_000001_object_center_overlay.png)

## Depth Metrics

- depth_mae_m: `0.002618759870529175`
- depth_rmse_m: `0.0030131845269352198`
- depth_error_median_m: `0.0027382373809814453`
- depth_error_p90_m: `0.004108905792236328`
- depth_error_p95_m: `0.004889130592346184`
- pred_depth_valid_ratio: `1.0`

## Point Clouds

- pointcloud_dir: `/home1/banghai/IsaacLab/source/standalone/stereo_spatial_calibration/outputs/pointclouds`
