from __future__ import annotations

import numpy as np

from stereo_spatial_calib.segmentation_utils import apply_mask_morphology, apply_mask_shift, get_mask_center_pixel


def test_mask_shift_moves_center_pixel():
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:6, 3:6] = True
    before = get_mask_center_pixel(mask)
    shifted = apply_mask_shift(mask, shift_x_px=2, shift_y_px=-1)
    after = get_mask_center_pixel(shifted)
    np.testing.assert_allclose(after, [before[0] + 2, before[1] - 1])


def test_mask_morphology_erode_and_dilate_change_area():
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True
    eroded = apply_mask_morphology(mask, mode="erode", kernel_size=3)
    dilated = apply_mask_morphology(mask, mode="dilate", kernel_size=3)
    assert eroded.sum() < mask.sum()
    assert dilated.sum() > mask.sum()

