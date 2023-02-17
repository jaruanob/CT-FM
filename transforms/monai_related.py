from typing import Dict, List

import torch
from monai.transforms import Transform


class ExtractFromDict(Transform):
    def __init__(self, keys: str) -> None:
        super().__init__()
        self.keys = keys

    def __call__(self, data: Dict[str, torch.Tensor]) -> List:
        return [data[key] for key in self.keys]
