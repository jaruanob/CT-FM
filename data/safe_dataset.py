import traceback

import torch
from torch.utils.data import Dataset
from loguru import logger


class SafeDataset(Dataset):
    """
    A wrapper around a Dataset that patches the __getitem__ method to catch exceptions
    and return None if an exception occurs, assuming that it happened because the file 
    is corrupted or missing and not because of a bug in the code.

    Attributes:
        dataset (Dataset): The original PyTorch Dataset to be wrapped.
        disable (bool): If True, disables the wrapper's try-except.
    """

    def __init__(self, dataset: Dataset, disable: bool = False):
        self.dataset = dataset
        self.disable = disable

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves an item from the wrapped dataset at the given index.
        If an exception occurs, logs the error and returns None.

        Args:
            index (int): The index of the item to be retrieved.

        Returns:
            Any: The item from the wrapped dataset at the given index, or None if an exception occurs.
        """
        if self.disable:
            return self.dataset[index]
        try:
            return self.dataset[index]
        except Exception as e:
            logger.error(f"Error at index {index}, skipping it. \nException: {e}\n{traceback.format_exc()}")
            return None
