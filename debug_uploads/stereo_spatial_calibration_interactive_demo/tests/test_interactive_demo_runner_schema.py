from stereo_spatial_calib.interactive_demo_runner import _write_interactive_report


def test_interactive_demo_result_schema(tmp_path):
    result = {
        "status": "success",
        "depth_source": "foundation_stereo_depth",
        "mask_source": "native_instance",
        "pointcloud_reference_view": "left",
        "center_pred_left_camera_m": [0.0, 0.0, 1.0],
        "center_pred_world_m": [0.5, 0.0, 0.04],
        "center_pred_base_m": [0.5, 0.0, 0.04],
        "center_gt_world_m": [0.5, 0.0, 0.04],
        "error_l2_mm": 0.0,
        "baseline_expected_m": 0.1,
        "baseline_actual_m": 0.1,
        "baseline_error_m": 0.0,
        "rotation_error_deg": 0.0,
    }
    _write_interactive_report(tmp_path, result)
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.html").exists()
    assert "pointcloud_reference_view" in (tmp_path / "report.md").read_text()

