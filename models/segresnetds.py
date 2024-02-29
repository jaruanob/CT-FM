from torch import nn
import monai


class SegResNetDSEncoder(monai.networks.nets.SegResNetDS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove the decoder
        del self.up_layers

        if self.spatial_dims == 2:
            self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif self.spatial_dims == 3:
            self.average_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        else:
            raise ValueError("`spatial_dims` must be 2 or 3.")
        
    def forward(self, x):
        # Return the last encoder layer's output,
        # the previous layer's output were there for skip connections
        x = self.encoder(x)[-1]
        x = self.average_pool(x)
        return x
