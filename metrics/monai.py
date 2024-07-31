from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from monai.metrics import compute_generalized_dice
from torch import Tensor
import torch

class GeneralizedDiceScore(Metric):
    def __init__(self, include_background: bool = False, weight_type: str = 'uniform', per_class: bool = False):
        super().__init__()
        self.include_background = include_background
        self.weight_type = weight_type
        self.per_class = per_class
        self.add_state("dice", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        sum_over_labels = not self.per_class
        result = compute_generalized_dice(preds, target, include_background=self.include_background, weight_type=self.weight_type, sum_over_labels=sum_over_labels)
        self.dice.append(result.mean(0) if self.per_class else result.mean())

    def compute(self) -> Tensor:
        dice = dim_zero_cat(self.dice)
        return dice
