import numpy as np

from stereo_spatial_calib.native_segmentation import ordered_mask_sources, select_mask_from_frame


def test_mask_source_auto_priority():
    assert ordered_mask_sources("auto") == [
        "native_instance",
        "native_instance_id",
        "native_semantic",
        "native_bbox_2d_tight",
        "projected_bbox_fallback",
    ]


def test_auto_selects_instance_before_semantic():
    inst = np.zeros((4, 4), dtype=np.int32)
    inst[1:3, 1:3] = 7
    sem = np.ones((4, 4), dtype=np.int32)
    frame = {
        "instance_segmentation": {"data": inst, "info": {"idToLabels": {"7": "target_cube"}}},
        "semantic_segmentation": {"data": sem},
    }
    selected = select_mask_from_frame(frame, inst.shape, "auto", target_names=("target_cube",))
    assert selected.mask_source == "native_instance"
    assert selected.target_instance_id == 7
    assert selected.mask.sum() == 4
