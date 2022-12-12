from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from loguru import logger


class NLSTDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, mode, transform, prototyping_num_scans=None):

        assert mode in ["train", "tune", "test"]
        self.mode = mode
        self.root_dir = Path(root_dir)
        split = get_dataset_split(self.root_dir / "SelectionTrainTestFinal.csv")
        self.scan_paths = [self.root_dir / "nrrd" / image for image in split[mode]]
        self.scan_paths = sorted(list(filter(lambda path: path.is_file(), self.scan_paths)))

        # Reduce the number of datapoints for prototyping
        if prototyping_num_scans is not None:
            self.scan_paths = self.scan_paths[:prototyping_num_scans]

        self.transform = transform

        logger.info(f"{mode.capitalize()} dataset has {len(self)} datapoints")

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


def get_dataset_split(split_csv):
    split_df = pd.read_csv(split_csv)
    split = {"train": [], "tune": [], "test": []}
    for split_name in split:
        for timepoint_name in ["T0", "T1", "T2"]:
            temp_df = split_df.loc[split_df.Data_Set == split_name.capitalize()]
            temp_df = temp_df.loc[temp_df[timepoint_name] == 1]
            scans = [f"{timepoint_name}/{id}_img.nrrd" for id in list(temp_df.Patient_ID)]
            split[split_name].extend(scans)
    return split