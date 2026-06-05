"""Reserved adapter for GroundingDINO + SAM / YOLO-seg style segmentation."""

from __future__ import annotations


class OpenVocabSegmentationAdapter:
    """TODO: replace Isaac GT masks with open-vocabulary segmentation masks."""

    def predict_mask(self, image_rgb, text_prompt):
        raise NotImplementedError("Open-vocabulary segmentation is reserved for the next version.")

