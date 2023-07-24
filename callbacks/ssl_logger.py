from typing import Any, Dict

import torch
from lighter.callbacks import LighterLogger
from pytorch_lightning import Trainer


def map_embedding_to_hypersphere_bins(embeddings):
    """
    This function maps input embeddings to bins on a unit hypersphere and returns the bin indices
    sorted based on their frequency.

    Args:
        embeddings (torch.Tensor): A tensor of shape (batch_size, embedding_dims) containing input
                                   embeddings.

    Returns:
        sorted_closest_sample_idx (torch.Tensor): A tensor of shape (batch_size,) containing the
                                                  sorted bin indices corresponding to the input
                                                  embeddings.

    Procedure:
        1. Normalize the input embeddings to get their projection on the unit sphere.
        2. Sample 500 uniform bins on the unit sphere.
        3. Compute angles between the normalized input embeddings and the bins.
        4. Get the index of the closest bin for each input embedding.
        5. Count the unique bin indices and their frequencies.
        6. Sort the unique bin indices based on their frequencies.
        7. Create a mapping from the old bin indices to the new indices based on frequency order.
        8. Generate the final sorted bin indices tensor using the index mapping.
    """

    # Normalize the input embeddings to get their projection on the unit sphere
    unit_vectors = embeddings / embeddings.norm(dim=-1, keepdim=True)

    # Sample 500 uniform bins on the unit sphere
    embedding_dims = embeddings.shape[-1]
    unit_sphere_samples = torch.randn(500, embedding_dims, dtype=embeddings.dtype)
    unit_sphere_samples = unit_sphere_samples / unit_sphere_samples.norm(
        dim=-1, keepdim=True
    )

    # Transfer the unit sphere samples to the same device as the input embeddings
    unit_sphere_samples = unit_sphere_samples.to(embeddings.device)

    # Compute angles between the normalized input embeddings and the bins
    angles = torch.acos(unit_vectors @ unit_sphere_samples.T)

    # Get the index of the closest bin for each input embedding
    closest_sample_idx = torch.argmin(angles, dim=1)

    # Count the unique bin indices and their frequencies
    unique_elements, counts = torch.unique(closest_sample_idx, return_counts=True)

    # Sort the unique bin indices based on their frequencies
    sorted_indices = torch.argsort(counts, descending=True)

    # Create a mapping from the old bin indices to the new indices based on frequency order
    index_mapping = torch.zeros_like(unique_elements)
    index_mapping[sorted_indices] = torch.arange(len(unique_elements))

    # Generate the final sorted bin indices tensor using the index mapping
    sorted_closest_sample_idx = index_mapping[closest_sample_idx]

    return sorted_closest_sample_idx.detach().cpu()


class SSLLogger(LighterLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_by_type(
        self, log_type: str, name: str, data: Any, data_name: str, global_step: int
    ) -> None:
        if log_type == "embedding_histogram" and data_name == "pred":
            embeddings = self._recursive_concat(data)
            # Map embeddings to bins
            hypersphere_bin_distribution = map_embedding_to_hypersphere_bins(embeddings)

            super().log_by_type(
                log_type="histogram",
                name=f"{name}/hypersphere_bin_distribution",
                data=hypersphere_bin_distribution,
                data_name="pred",
                global_step=global_step
            )
        else:
            super().log_by_type(log_type, name, data, data_name, global_step)

        def _on_batch_end(self, outputs: Dict, trainer: Trainer) -> None:
            if not trainer.sanity_checking:
                outputs["pred"] = self._recursive_all_gather(outputs["pred"], trainer)

            super()._on_batch_end(outputs, trainer)

        def _recursive_concat(self, data):
            if isinstance(data, dict):
                embeddings = []
                for _, value in data.items():
                    embeddings.append(self._recursive_concat(value))
                return torch.cat(embeddings)
            elif isinstance(data, (list, tuple)):
                embeddings = [self._recursive_concat(value) for value in data]
                return torch.cat(embeddings)
            else:
                return data

        def _recursive_all_gather(self, data, trainer):
            if isinstance(data, dict):
                for key, value in data.items():
                    data[key] = self._gather_all_data(value, trainer)
            elif isinstance(data, (list, tuple)):
                data = list(data)
                for idx, value in enumerate(data):
                    data[idx] = self._gather_all_data(value, trainer)
            else:
                data = trainer.strategy.all_gather(data).reshape(-1, *data.shape[1:])
            return data
