from torch import nn
import torch
from typing import Union, List
import monai


class SegResNetDSwEmbedding(monai.networks.nets.SegResNetDS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.spatial_dims == 2:
            self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif self.spatial_dims == 3:
            self.average_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        else:
            raise ValueError("`spatial_dims` must be 2 or 3.")
        
    def _forward(self, x: torch.Tensor) -> Union[None, torch.Tensor, List[torch.Tensor]]:
        if self.preprocess is not None:
            x = self.preprocess(x)

        if not self.is_valid_shape(x):
            raise ValueError(f"Input spatial dims {x.shape} must be divisible by {self.shape_factor()}")

        x_down = self.encoder(x)

        x_down.reverse()
        x = x_down.pop(0)
        embedding = self.average_pool(x).flatten(start_dim=1)

        if len(x_down) == 0:
            x_down = [torch.zeros(1, device=x.device, dtype=x.dtype)]

        outputs: list[torch.Tensor] = []

        i = 0
        for level in self.up_layers:
            x = level["upsample"](x)
            x += x_down.pop(0)
            x = level["blocks"](x)

            if len(self.up_layers) - i <= self.dsdepth:
                outputs.append(level["head"](x))
            i = i + 1

        outputs.reverse()

        # in eval() mode, always return a single final output
        if not self.training or len(outputs) == 1:
            return outputs[0]

        # return a list of DS outputs
        return embedding, outputs


