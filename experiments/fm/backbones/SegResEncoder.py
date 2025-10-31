
import monai
import torch

def load_model(path, spatial_dims, in_channels, init_filters, blocks_down, head_module):
    model = monai.networks.nets.segresnet_ds.SegResEncoder(
        spatial_dims=spatial_dims, 
        in_channels=in_channels, 
        init_filters=init_filters, 
        blocks_down=blocks_down, 
        head_module=head_module
    )    
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt.state_dict(),strict=False)
    return model