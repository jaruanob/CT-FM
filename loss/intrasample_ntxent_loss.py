from typing import List
import torch
from torch import nn
from torch import distributed as torch_dist


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
            raise ValueError(
                "Illegal temperature: abs({}) < 1e-8".format(self.temperature)
            )
        

    def forward(self, out: tuple):
        """
        The forward pass of the IntraSampleNTXEntLoss.


        Args:
            out (tuple of tuple of Tensors): A tuple of input data. 
            Each element of the tuple represents a sample drawn from an image (dataset item)
            Each sample is a tuple of Tensors representing different views of the same image.
            Each view is a Tensor of shape (batch_size, channels, height, width).

        Returns:
            float: Contrastive Cross Entropy Loss value.
        """

        device = out[-1][-1].device
        batch_size = out[-1][-1].shape[0]

        overall_loss = 0
        for b in range(batch_size):
            per_sample_loss = 0
            for s_i, sample in enumerate(out):
                pos0 = sample[0][b].unsqueeze(0)
                pos0 = nn.functional.normalize(pos0, dim=1)

                pos1 = sample[1][b].unsqueeze(0)
                pos1 = nn.functional.normalize(pos1, dim=1)
                
                negatives = [out[s_j][0][b] for s_j, _ in enumerate(out) if s_j != s_i]
                negatives.extend([out[s_j][1][b] for s_j, _  in enumerate(out) if s_j != s_i])

                negatives = torch.stack(negatives, dim=0)
                negatives = nn.functional.normalize(negatives, dim=1)

                sim_pos = torch.einsum("nc,nc->n", [pos0, pos1]).unsqueeze(-1)
                sim_neg = torch.einsum("nc,kc->nk", [pos0, negatives])

                logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

                loss = self.cross_entropy(logits, labels)

                per_sample_loss += loss

            per_sample_loss /= len(out)
            overall_loss += per_sample_loss

        overall_loss /= batch_size

        return overall_loss
