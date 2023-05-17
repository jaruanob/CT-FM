
from torch import nn
from lightly.models.modules.heads import BarlowTwinsProjectionHead, VicRegLLocalProjectionHead


class VICRegL(nn.Module):
    def __init__(self, backbone, spatial_dims, num_ftrs, global_projection_ftrs=(2048, 2048), local_projection_ftrs=(128, 128)):
        super().__init__()
        self.backbone = backbone
        self.spatial_dims = spatial_dims
        self.global_projection_head = BarlowTwinsProjectionHead(num_ftrs, *global_projection_ftrs)
        self.local_projection_head = VicRegLLocalProjectionHead(num_ftrs, *local_projection_ftrs)
        if self.spatial_dims == 2:
            self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif self.spatial_dims == 3:
            self.average_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        else:
            raise ValueError("`spatial_dims` must be 2 or 3.")

    def forward(self, input):
        high_resolution, low_resolution = input

        global_views_features = [self.subforward(data["image"]) for data in high_resolution]
        global_grids = [data["grid"] for data in high_resolution]

        local_views_features = [self.subforward(data["image"]) for data in low_resolution]
        local_grids = [data["grid"] for data in low_resolution]

        return global_views_features, global_grids, local_views_features, local_grids

    def subforward(self, x):
        x = self.backbone(x)
        # Global features
        y_global = self.average_pool(x).flatten(start_dim=1)
        z_global = self.global_projection_head(y_global)

        # Local features
        dims_after_batch_and_channel = list(range(2, x.ndim))
        y_local = x.permute(0, *dims_after_batch_and_channel, 1)
        z_local = self.local_projection_head(y_local)

        return z_global, z_local
