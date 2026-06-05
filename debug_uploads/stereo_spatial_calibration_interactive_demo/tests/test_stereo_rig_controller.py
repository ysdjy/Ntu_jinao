import numpy as np

from stereo_spatial_calib.stereo_rig_controller import StereoRigController


def test_right_camera_local_position_equals_baseline():
    rig = StereoRigController().create_rig(baseline_m=0.10)
    assert np.allclose(rig._right_local_position, [0.10, 0.0, 0.0])
    assert rig.get_left_camera_path() == "/World/StereoRig/left_camera"
    assert rig.get_right_camera_path() == "/World/StereoRig/right_camera"


def test_rig_pose_roundtrip():
    rig = StereoRigController()
    rig.set_rig_world_pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    pos, quat = rig.get_rig_world_pose()
    assert np.allclose(pos, [1.0, 2.0, 3.0])
    assert np.allclose(quat, [0.0, 0.0, 0.0, 1.0])

