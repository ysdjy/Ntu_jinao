from stereo_spatial_calib.stereo_rig_controller import StereoRigController


def test_baseline_validation_correct_after_create_and_repair():
    rig = StereoRigController().create_rig(baseline_m=0.125)
    status = rig.validate_baseline_and_orientation()
    assert abs(status["baseline_actual_m"] - 0.125) < 1e-9
    assert abs(status["baseline_error_m"]) < 1e-9
    assert status["rotation_error_deg"] == 0.0
    assert status["is_rectified_like"] is True

    rig._right_local_position[0] = 0.2
    assert rig.validate_baseline_and_orientation()["is_rectified_like"] is False
    repaired = rig.repair_rig()
    assert repaired["is_rectified_like"] is True

