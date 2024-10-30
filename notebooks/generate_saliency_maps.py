import gc
import os
import random
import monai
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from monai.networks.nets import SegResNetDS
from monai.visualize import OcclusionSensitivity
from torch.nn.functional import cosine_similarity
from utils import adjust_prefix_and_load_state_dict

# Data paths and setup
image_dir = Path("/mnt/data6/ibro/Datasets/Dataset606_TotalSegmentator/imagesTr")
label_dir = Path("/mnt/data6/ibro/Datasets/Dataset606_TotalSegmentator/labelsTr")
images = list(image_dir.glob("*.nii.gz"))

# Preprocessing transforms
keys = ["image", "label"]
image_key = "image"
transforms = monai.transforms.Compose([
    monai.transforms.LoadImaged(keys=keys, ensure_channel_first=True),
    monai.transforms.EnsureTyped(keys=keys),
    monai.transforms.Orientationd(keys=keys, axcodes="SPL"),
    monai.transforms.Spacingd(keys=keys, pixdim=[3,3,3], mode="bilinear"),
    monai.transforms.CropForegroundd(keys=keys, source_key=image_key),
    monai.transforms.ScaleIntensityRanged(keys=image_key, a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
    monai.transforms.Lambda(func=lambda x: [x[k].as_tensor() for k in keys])
])

# Process images
random.shuffle(images)
images = [transforms({"image": image, "label": str(image).replace("imagesTr", "labelsTr").replace("_0000", "")}) 
          for image in images[:10]]

class EmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = adjust_prefix_and_load_state_dict(
            ckpt_path="/mnt/data16/ibro/IDC_SSL_CT/runs/fm/checkpoints/CT_FM_SimCLR_SegResNetDS/epoch=449-step=225000-v1.ckpt",
            ckpt_to_model_prefix={"backbone." : "encoder"},
            model=SegResNetDS(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                init_filters=32,
                blocks_down=[1, 2, 2, 4, 4],
                dsdepth=4
            )
        )
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = torch.nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.model.encoder(x)
        x.reverse()
        x = x.pop(0)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x


# Setup model and occlusion sensitivity
encoder = EmbeddingModel().to("cuda:0")
occ_sens = OcclusionSensitivity(nn_module=encoder, n_batch=12, activate=False, mask_size=10, overlap=0.25)

# Generate distance maps
distance_maps = []
for x in images:
    x = x[0].unsqueeze(0).to("cuda:0")
    occ_map, _ = occ_sens(x)

    encoder.eval()
    with torch.no_grad():
        base_embedding = encoder(x)
        distances = 1 - cosine_similarity(
            base_embedding.flatten().unsqueeze(0),
            occ_map.view(512, -1).t(),
            dim=1
        ).squeeze()
        
        distances = distances.view(*occ_map.shape[2:])
        distance_maps.append(distances.cpu())

        torch.cuda.empty_cache()
        gc.collect()

# Save results
torch.save({
    'distance_maps': distance_maps,
    'original_images': images
}, "saliency_data.torch")
