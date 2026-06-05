from __future__ import annotations

import numpy as np

from stereo_spatial_calib.native_capture_methods import NativeCaptureResult


def test_native_capture_result_schema_holds_arrays_and_metadata():
    result = NativeCaptureResult(
        left_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        right_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        gt_depth_left=np.ones((480, 640), dtype=np.float32),
        gt_instance_mask=np.ones((480, 640), dtype=np.uint8),
        gt_segmentation=None,
        metadata={"sample_id": "cube_000001"},
        capture_method="camera_class",
        mask_source="projected_bbox_fallback",
    )
    assert result.left_rgb.shape == (480, 640, 3)
    assert result.gt_depth_left.shape == result.gt_instance_mask.shape
    assert result.capture_method == "camera_class"
    assert result.mask_source == "projected_bbox_fallback"

