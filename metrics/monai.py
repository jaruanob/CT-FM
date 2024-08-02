from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from monai.metrics import compute_generalized_dice
from torch import Tensor

class GeneralizedDiceScore(Metric):
    """
    Computes the Generalized Dice Score for segmentation tasks.
    
    Args:
        include_background (bool): Whether to include the background class in the computation.
        weight_type (str): Type of weighting to use ('uniform', 'volume', etc.).
        per_class (bool): Whether to compute the score per class or as a single value.
    """
    def __init__(self, include_background: bool = False, weight_type: str = 'uniform', per_class: bool = False):
        super().__init__()
        self.include_background = include_background
        self.weight_type = weight_type
        self.per_class = per_class
        self.add_state("dice", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the state with predictions and targets.
        
        Args:
            preds (Tensor): Predicted tensor.
            target (Tensor): Ground truth tensor.
        """
        sum_over_labels = not self.per_class
        result = compute_generalized_dice(preds, target, include_background=self.include_background, weight_type=self.weight_type, sum_over_labels=sum_over_labels)
        self.dice.append(result.mean(0) if self.per_class else result.mean())

    def compute(self) -> Tensor:
        """
        Compute the final Generalized Dice Score.
        
        Returns:
            Tensor: The computed Generalized Dice Score.
        """
        return dim_zero_cat(self.dice)