import torch


class IDCDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, transform):
        self.image_files = sorted(image_files)
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.image_files[idx]), None

    def __len__(self):
        return len(self.image_files)
