import numpy as np


def test_pointcloud_reference_view_metadata_aliases():
    T = np.eye(4, dtype=np.float32).tolist()
    metadata = {
        "pointcloud_reference_view": "left",
        "T_world_left_camera": T,
        "T_world_left_cam": T,
        "T_world_reference_camera": T,
    }
    assert metadata["pointcloud_reference_view"] == "left"
    assert metadata["T_world_left_camera"] == metadata["T_world_reference_camera"]


def test_center_pred_camera_compatibility_alias():
    row = {"center_pred_camera_m": [1.0, 2.0, 3.0]}
    row.setdefault("center_pred_left_camera_m", row["center_pred_camera_m"])
    assert row["center_pred_left_camera_m"] == row["center_pred_camera_m"]

