import torch
import torch.nn as nn

from .mednext_blocks import *

class MedNeXt(nn.Module):
    """
    MedNeXt model class.

    Args:
        init_filters (int): Number of initial filters.
        in_channels (int): Number of input channels.
        n_classes (int): Number of output classes.
        enc_exp_r (int, optional): Expansion ratio for encoder blocks. Defaults to 2.
        dec_expr_r (int, optional): Expansion ratio for decoder blocks. Defaults to 2.
        bottlenec_exp_r (int, optional): Expansion ratio for bottleneck blocks. Defaults to 2.
        kernel_size (int, optional): Kernel size for convolutions. Defaults to 7.
        deep_supervision (bool, optional): Whether to use deep supervision. Defaults to False.
        do_res (bool, optional): Whether to use residual connections. Defaults to False.
        do_res_up_down (bool, optional): Whether to use residual connections in up and down blocks. Defaults to False.
        enc_blocks (list, optional): Number of blocks in each encoder stage. Defaults to [2, 2, 2, 2].
        bottleneck_blocks (list, optional): Number of blocks in bottleneck stage. Defaults to 2.
        dec_blocks (list, optional): Number of blocks in each decoder stage. Defaults to [2, 2, 2, 2].
        norm_type (str, optional): Type of normalization layer. Defaults to 'group'.
        dim (str, optional): Dimension of the model ('2d' or '3d'). Defaults to '3d'.
        grn (bool, optional): Whether to use Global Response Normalization (GRN). Defaults to False.
    """
    def __init__(self, 
        init_filters: int, 
        in_channels: int,
        n_classes: int, 
        enc_exp_r: int = 2,
        dec_expr_r: int = 2,
        bottlenec_exp_r: int = 2,
        kernel_size: int = 7,
        deep_supervision: bool = False,
        do_res: bool = False,
        do_res_up_down: bool = False,
        enc_blocks: list = [2, 2, 2, 2],  
        bottleneck_blocks: list = 2,
        dec_blocks: list = [2, 2, 2, 2],
        norm_type = 'group',
        dim = '3d',
        grn = False
    ):
        """
        Initialize the MedNeXt model.

        This method sets up the architecture of the model, including:
        - Stem convolution
        - Encoder stages and downsampling blocks
        - Bottleneck blocks
        - Decoder stages and upsampling blocks
        - Output blocks for deep supervision (if enabled)
        """
        super().__init__()

        self.do_ds = deep_supervision
        assert dim in ['2d', '3d'], "dim must be '2d' or '3d'"
        enc_kernel_size = dec_kernel_size = kernel_size

        if isinstance(enc_exp_r, int):
            enc_exp_r = [enc_exp_r] * len(enc_blocks)

        if isinstance(dec_expr_r, int):
            dec_expr_r = [dec_expr_r] * len(dec_blocks)

        conv = nn.Conv2d if dim == '2d' else nn.Conv3d
            
        self.stem = conv(in_channels, init_filters, kernel_size=1)
        
        enc_stages = []
        down_blocks = []

        for i, num_blocks in enumerate(enc_blocks):
            enc_stages.append(nn.Sequential(*[
                MedNeXtBlock(
                    in_channels=init_filters * (2 ** i),
                    out_channels=init_filters * (2 ** i),
                    exp_r=enc_exp_r[i],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn
                ) 
                for _ in range(num_blocks)]
            ))
            
            down_blocks.append(MedNeXtDownBlock(
                in_channels=init_filters * (2 ** i),
                out_channels=init_filters * (2 ** (i + 1)),
                exp_r=enc_exp_r[i],
                kernel_size=enc_kernel_size,
                do_res=do_res_up_down,
                norm_type=norm_type,
                dim=dim
            ))
    
        self.enc_stages = nn.ModuleList(enc_stages)
        self.down_blocks = nn.ModuleList(down_blocks)

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=init_filters * (2 ** len(enc_blocks)),
                out_channels=init_filters * (2 ** len(enc_blocks)),
                exp_r=bottlenec_exp_r,
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for _ in range(bottleneck_blocks)]
        )
        
        up_blocks = []
        dec_stages = []
        for i, num_blocks in enumerate(dec_blocks):
            up_blocks.append(MedNeXtUpBlock(
                in_channels=init_filters * (2 ** (len(dec_blocks) - i)),
                out_channels=init_filters * (2 ** (len(dec_blocks) - i - 1)),
                exp_r=dec_expr_r[i],
                kernel_size=dec_kernel_size,
                do_res=do_res_up_down,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            ))
            
            dec_stages.append(nn.Sequential(*[
                MedNeXtBlock(
                    in_channels=init_filters * (2 ** (len(dec_blocks) - i - 1)),
                    out_channels=init_filters * (2 ** (len(dec_blocks) - i - 1)),
                    exp_r=dec_expr_r[i],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn
                )
                for _ in range(num_blocks)]
            ))
            
        self.up_blocks = nn.ModuleList(up_blocks)
        self.dec_stages = nn.ModuleList(dec_stages)

        self.out_0 = OutBlock(in_channels=init_filters, n_classes=n_classes, dim=dim)

        if deep_supervision:
            self.out_blocks = nn.ModuleList([
                OutBlock(in_channels=init_filters * (2 ** i), n_classes=n_classes, dim=dim) 
                for i in range(1, len(enc_blocks))
            ])

    def forward(self, x):
        """
        Forward pass of the MedNeXt model.

        This method performs the forward pass through the model, including:
        - Stem convolution
        - Encoder stages and downsampling
        - Bottleneck blocks
        - Decoder stages and upsampling with skip connections
        - Output blocks for deep supervision (if enabled)

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor or list: Output tensor(s).
        """
        # Apply stem convolution
        x = self.stem(x)
        
        # Encoder forward pass
        enc_outputs = []
        for enc_stage, down_block in zip(self.enc_stages, self.down_blocks):
            x = enc_stage(x)
            enc_outputs.append(x)
            x = down_block(x)
        
        # Bottleneck forward pass
        x = self.bottleneck(x)
        
        # Initialize deep supervision outputs if enabled
        if self.do_ds:
            ds_outputs = [self.out_blocks[-1](x)]
        
        # Decoder forward pass with skip connections
        for i, (up_block, dec_stage) in enumerate(zip(self.up_blocks, self.dec_stages)):
            x = up_block(x)
            x = x + enc_outputs[-(i + 1)]
            x = dec_stage(x)
            if self.do_ds:
                ds_outputs.append(self.out_blocks[-(i + 1)](x))
        
        # Final output block
        x = self.out_0(x)
        
        # Return output(s)
        if self.do_ds:
            return [x] + list(reversed(ds_outputs))
        else:
            return x