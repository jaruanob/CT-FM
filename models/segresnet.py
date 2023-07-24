from torch import nn
import monai


class SegResNetForPretraining(monai.networks.nets.SegResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove the decoder layers
        del self.up_layers, self.up_samples, self.conv_final

    def forward(self, x):
        x, down_x = self.encode(x)
        return x
