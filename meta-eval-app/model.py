import torch
import os
from monai.networks.nets import SegResNetDS
from utils import adjust_prefix_and_load_state_dict
import streamlit as st

# Wrap the segresnet model in a module that returns the embeddings
class EmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = adjust_prefix_and_load_state_dict(
                    ckpt_path="/mnt/data1/CT_FM/IDC_SSL_CT/runs/checkpoints/CT_FM_ConRecon_SegResNetDS/epoch=79-step=40000.ckpt",
                    ckpt_to_model_prefix={"backbone." : ""},
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
    
def load_model():
    model = EmbeddingModel()
    model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.eval()
    return model
