import torch
import os
# from monai.networks.nets.segresnet_ds import SegResNetEncoder
from utils import adjust_prefix_and_load_state_dict
import streamlit as st
import monai

# Wrap the segresnet model in a module that returns the embeddings
class EmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = adjust_prefix_and_load_state_dict(
                    ckpt_path="/mnt/data1/CT_FM/latest_fm_checkpoints/original/epoch=449-step=225000-v1.ckpt",
                    ckpt_to_model_prefix={"backbone.": ""},
                    model=monai.networks.nets.segresnet_ds.SegResEncoder(
                        spatial_dims=3,
                        in_channels=1,
                        init_filters=32,
                        blocks_down=[1, 2, 2, 4, 4],
                        head_module=lambda x: x[-1]
                    ),
                    
            )
        
        # import sys
        # sys.path.append('/home/suraj/Repositories/lighter-ct-fm')

        # from models.suprem import SuPreM_loader
        # from models.backbones.unet3d import UNet3D

        # self.model = SuPreM_loader(
        #     model=UNet3D(
        #         n_class=10
        #     ),
        #     ckpt_path="/mnt/data1/CT_FM/baselines/SuPreM_UNet/supervised_suprem_unet_2100.pth",
        #     decoder=False,
        #     encoder_only=True
        # )

        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = torch.nn.Flatten(start_dim=1)

    def forward(self, x):
        x = x.permute(0, 1, 4, 3, 2)
        x = x.flip(2).flip(3)
        x = self.model(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x
    
def load_model():
    model = EmbeddingModel()
    model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.eval()
    return model
