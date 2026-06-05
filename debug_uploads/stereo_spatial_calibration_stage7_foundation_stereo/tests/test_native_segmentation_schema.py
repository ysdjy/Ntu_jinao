import numpy as np

from stereo_spatial_calib.native_segmentation import extract_id_map, is_colorized_segmentation, mask_from_id_map, select_target_id


def test_id_map_generates_binary_mask_for_target_id():
    id_map = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint32)
    target_id = select_target_id(id_map, labels={"2": "target_cube"}, target_names=("target_cube",))
    mask = mask_from_id_map(id_map, target_id)
    assert target_id == 2
    assert mask.dtype == np.bool_
    assert mask.sum() == 2


def test_colorized_segmentation_is_not_id_map():
    color = np.zeros((4, 4, 4), dtype=np.uint8)
    color[..., 0] = np.arange(4, dtype=np.uint8).reshape(4, 1)
    color[..., 3] = 255
    assert is_colorized_segmentation(color)
    assert extract_id_map({"data": color}) is None
