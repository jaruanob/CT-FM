from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from loguru import logger


class IDCDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform, prototyping_num_scans=None):
        self.scan_paths = sorted(list(Path(root_dir).glob("*.nrrd")))

        # Reduce the number of datapoints for prototyping
        if prototyping_num_scans is not None:
            self.scan_paths = self.scan_paths[:prototyping_num_scans]

        self.transform = transform


    def __getitem__(self, idx):
        scan_path = self.scan_paths[idx]
        # Load the scan
        try:
            scan = sitk.ReadImage(str(scan_path))
        except RuntimeError:
            logger.info(f"Failed to load scan {scan_path}, skipping it.")
            return None

        label = 0  # Dummy label, no labels used in SSL training
        tensor = self.transform(scan)
        if isinstance(tensor, (tuple, list)):
            return *tensor, label
        return tensor, label

    def __len__(self):
        return len(self.scan_paths)
