import torch
import torch.nn as nn


class Reconstruction(nn.Module):
    """
    This class wraps the model so that the format with `backbone` is the same as the other frameworks available.
    No extra functionality, just the backbone's forward is called.
    """

    def __init__(self, backbone: nn.Module):
        """
        Args:
            backbone (nn.Module): The backbone model to be wrapped.
        """
        super().__init__()
        self.backbone = backbone

    def forward(self, input):
        return self.backbone(input)