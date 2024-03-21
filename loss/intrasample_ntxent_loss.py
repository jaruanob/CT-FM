import torch
from torch import nn


class IntraSampleNTXEntLoss(nn.Module):
    """
    This class implements the IntraSampleNTXEntLoss, a loss function used for contrastive learning.
    """

    def __init__(self, temperature: float = 0.1):
        """
        Initialize an instance of the class.

        Args:
            temperature (float, optional): The temperature parameter for the instance. Defaults to 0.1.
        """
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8

        if abs(self.temperature) < self.eps:
            raise ValueError("Illegal temperature: abs({}) < 1e-8".format(self.temperature))

    def forward(self, input):
        """
        The forward pass of the IntraSampleNTXEntLoss.

        Args:
            input (list of list of Tensors): The input data in the following format:
                List [crops], List [views], Tensor [B, out_dim].
                Each batch index represents a different image.
        Returns:
            float: Contrastive Cross Entropy Loss value.
        """
        # Dimensions: List [crops], List [views], Tensor [B, out_dim].
        batch_size = input[-1][-1].shape[0]
        num_crops = len(input)

        loss = 0
        for b in range(batch_size):
            per_sample_loss = 0
            for i in range(num_crops):
                # Positives: same crop (i), both views (0 and 1), from the same image (b - batch index)
                positive_0 = input[i][0][b].unsqueeze(0)
                positive_1 = input[i][1][b].unsqueeze(0)

                # Negatives: all other crops (j != i), both views (0 and 1), from the same image (b - batch index)
                negatives = []
                for j in range(num_crops):
                    if j != i:
                        negatives.append(input[j][0][b])
                        negatives.append(input[j][1][b])
                negatives = torch.stack(negatives, dim=0)

                positive_0 = nn.functional.normalize(positive_0, dim=1)
                positive_1 = nn.functional.normalize(positive_1, dim=1)
                negatives = nn.functional.normalize(negatives, dim=1)

                sim_pos = torch.einsum("nc,nc->n", positive_0, positive_1).unsqueeze(-1)
                sim_neg = torch.einsum("nc,kc->nk", positive_0, negatives)

                logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

                per_sample_loss += self.cross_entropy(logits, labels)
            loss += per_sample_loss / num_crops
        loss /= batch_size
        return loss
