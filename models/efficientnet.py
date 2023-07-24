from torch import nn
import monai



class EfficientNetForPretraining(monai.networks.nets.EfficientNetBN):
    def __init__(self, use_average_pooling=False, use_flattening=False, *args, **kwargs):
        self.use_average_pooling = use_average_pooling
        self.use_flattening = use_flattening
        super().__init__(*args, **kwargs)

    def forward(self, inputs):
        """
        """
        # Stem
        x = self._conv_stem(self._conv_stem_padding(inputs))
        x = self._swish(self._bn0(x))
        # Blocks
        x = self._blocks(x)
        # Head
        x = self._conv_head(self._conv_head_padding(x))
        x = self._swish(self._bn1(x))

        # ---- These layers are present in the original implementation but we don't pretrain them ----
        # # Pooling and final linear layer
        # x = self._avg_pooling(x)

        # x = x.flatten(start_dim=1)
        # x = self._dropout(x)
        # x = self._fc(x)
        # ---------------------------------------------------------------------------------------------

        # Added optional pooling [ibro]
        if self.use_average_pooling:
            x = self._avg_pooling(x)

        # Added optional flattening [ibro]
        if self.use_flattening:
            x = x.flatten(start_dim=1)

        return x

class EfficientNetWithFinalActivation(monai.networks.nets.efficientnet.EfficientNetBN):
    def __init__(self, final_activation=nn.Sigmoid(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_activation = final_activation

    def forward(self, inputs):
        return self.final_activation(super().forward(inputs))
