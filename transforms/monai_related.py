from typing import Callable, Dict, List, Union

import torch
from monai.transforms import Transform, BoundingRect, SpatialCropd
from monai.transforms.utils import  is_positive


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
