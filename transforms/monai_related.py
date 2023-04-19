from typing import Callable, Dict, List, Union

import torch
from monai.transforms import Transform, BoundingRect, SpatialCropd, RandSpatialCrop
from monai.transforms.utils import  is_positive

from loguru import logger


class ExtractFromDict(Transform):
    def __init__(self, keys: str) -> None:
        super().__init__()
        self.keys = keys

    def __call__(self, data: Dict[str, torch.Tensor]) -> List:
        return [data[key] for key in self.keys]


class SpatialCropAroundBoundingRectd(Transform):
    def __init__(
            self, crop_keys: Union[str, List[str]], bbox_key: str, select_fn: Callable = is_positive
        ) -> None:
        super().__init__()
        self.crop_keys = crop_keys
        self.bbox_key = bbox_key
        self.bbox_transform = BoundingRect(select_fn)

    def __call__(self, data) -> Dict:
        bbox = self.bbox_transform(data[self.bbox_key])
        roi_start = bbox[:len(bbox) // 2]
        roi_end = bbox[len(bbox) // 2:]
        crop_transform = SpatialCropd(keys=self.crop_keys, roi_start=roi_start, roi_end=roi_end)
        return crop_transform(data)

class FilteredRandSpatialCrop(RandSpatialCrop):
    def __init__(self, allowed_empty_voxel_proportion, roi_size, max_roi_size=None, random_center=True, random_size=True):
        super().__init__(roi_size=roi_size, max_roi_size=max_roi_size, random_center=random_center, random_size=random_size)
        self.allowed_empty_voxel_proportion = allowed_empty_voxel_proportion

    def __call__(self, img):
        cropped_img = super().__call__(img)
        counter = 0
        while not torch.count_nonzero(input=cropped_img) > self.allowed_empty_voxel_proportion * torch.numel(cropped_img):
            cropped_img = super().__call__(img)
            counter += 1
            if counter > 5:
                return cropped_img
                logger.info("Could not find a valid crop after 5 tries. Returning as-is.")
        return cropped_img
