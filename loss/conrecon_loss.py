from torch import nn


class ReconLossNestedWrapper(nn.Module):
    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, input, target):
        loss = 0
        for input_samples, target_samples in zip(input, target):
            for input_duplicate, target_duplicate in zip(input_samples, target_samples):
                loss += self.loss(input_duplicate, target_duplicate) 

        # Average the loss
        loss = loss / (len(input) * len(input[0]))
        return loss
        

class ConReconLoss(nn.Module):
    """
    This class implements the ConReconLoss, a loss function used for contrastive learning + reconstruction
    """

    def __init__(self, contrastive_loss: nn.Module, reconstruction_loss: nn.Module, contrastive_weight: float = 0.5):
        """
        Initialize an instance of the class.

        Args:
        """
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.reconstruction_loss = ReconLossNestedWrapper(reconstruction_loss)
        self.contrastive_weight = contrastive_weight        

    def forward(self, input, target):
        """
        The forward pass of the ConReconLoss.
        """
        assert "con" in input and "recon" in input, "input must contain both contrastive and reconstruction data"
        contrastive_loss = self.contrastive_loss(input["con"])
        reconstruction_loss = self.reconstruction_loss(input["recon"], target)

        return self.contrastive_weight * contrastive_loss + (1 - self.contrastive_weight) * reconstruction_loss