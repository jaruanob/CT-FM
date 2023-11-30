from torch import nn
import monai


class SegResNetDSEncoder(monai.networks.nets.SegResNetDS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove the decoder
        del self.up_layers

    def forward(self, x):
        # Return the last encoder layer's output,
        # the previous layer's output were there for skip connections
        return self.encoder(x)[-1]
