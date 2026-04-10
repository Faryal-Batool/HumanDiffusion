"""
U-Net backbone implementation for diffusion models.

This module provides a conditional 2D U-Net architecture designed for diffusion models,
featuring sinusoidal time embeddings, conditional residual blocks, and multi-scale
processing with skip connections. The architecture supports both local and global
conditioning for tasks like trajectory generation and image-to-image translation.
"""

import math
from typing import Union
import torch
import torch.nn as nn


# -----------------------------------------------------------
# Time embedding for diffusion timestep conditioning
# -----------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timesteps.

    This implements the sinusoidal position encoding similar to transformer models,
    but adapted for continuous timestep values in diffusion models. The embedding
    provides a way to encode timestep information that the model can use for
    conditioning.
    """
    def __init__(self, dim: int):
        """
        Initialize sinusoidal positional embedding.

        Args:
            dim: Dimensionality of the embedding output.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        """
        Generate sinusoidal embeddings for input timesteps.

        Args:
            x: Input tensor containing timestep values.

        Returns:
            torch.Tensor: Sinusoidal positional embeddings.
        """
        device = x.device
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# -----------------------------------------------------------
# 2D Convolutional Building Blocks
# -----------------------------------------------------------

class Conv2dBlock(nn.Module):
    """
    Basic 2D convolutional block with GroupNorm and Mish activation.

    This is a standard building block for U-Net architectures, providing
    convolution, normalization, and non-linear activation in a single module.
    """
    def __init__(self, inp_channels: int, out_channels: int,
                 kernel_size: int, n_groups: int = 8):
        """
        Initialize convolutional block.

        Args:
            inp_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            n_groups: Number of groups for GroupNorm.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation."""
        return self.block(x)


class Downsample2d(nn.Module):
    """
    2D downsampling block using strided convolution.

    Reduces spatial dimensions by factor of 2 while preserving channel count.
    """
    def __init__(self, dim: int):
        """
        Initialize downsampling block.

        Args:
            dim: Number of input/output channels (unchanged).
        """
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample input tensor by factor of 2."""
        return self.conv(x)


class Upsample2d(nn.Module):
    """
    2D upsampling block using transposed convolution.

    Increases spatial dimensions by factor of 2 while preserving channel count.
    """
    def __init__(self, dim: int):
        """
        Initialize upsampling block.

        Args:
            dim: Number of input/output channels (unchanged).
        """
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample input tensor by factor of 2."""
        return self.conv(x)


class ConditionalResidualBlock2D(nn.Module):
    """
    Conditional residual block for 2D features with adaptive normalization.

    This block implements adaptive normalization where conditioning information
    can control the scale and bias of feature maps. It supports both standard
    conditioning (adding to features) and scale-shift conditioning (FiLM-like).
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 cond_dim: int,
                 kernel_size: int = 3,
                 n_groups: int = 8,
                 cond_predict_scale: bool = False):
        """
        Initialize conditional residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            cond_dim: Dimensionality of conditioning input.
            kernel_size: Size of convolutional kernels.
            n_groups: Number of groups for GroupNorm.
            cond_predict_scale: If True, predict both scale and bias; else only bias.
        """
        super().__init__()

        # Two convolutional blocks for residual processing
        self.blocks = nn.ModuleList([
            Conv2dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv2dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # Conditioning encoder: maps condition to channel-wise modulation parameters
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels

        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        # Residual connection to handle channel dimension changes
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else
            nn.Identity()
        )

        # Initialize conditioning encoder weights
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.cond_encoder.apply(_init_weights)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditional modulation.

        Args:
            x: Input feature tensor (B, C, H, W).
            cond: Conditioning tensor (B, cond_dim).

        Returns:
            torch.Tensor: Output feature tensor with same spatial dims as input.
        """
        # First convolution block
        out = self.blocks[0](x)

        # Encode conditioning information
        embed = self.cond_encoder(cond)

        if self.cond_predict_scale:
            # Scale-shift conditioning: predict both scale and bias
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1, 1)
            scale = embed[:, 0, ...]  # (B, C, 1, 1)
            bias = embed[:, 1, ...]   # (B, C, 1, 1)
            out = scale * out + bias
        else:
            # Standard conditioning: add bias only
            embed = embed.view(embed.shape[0], self.out_channels, 1, 1)
            out = out + embed

        # Second convolution block
        out = self.blocks[1](out)

        # Residual connection
        out = out + self.residual_conv(x)
        return out


# -----------------------------------------------------------
# 2D Conditional U-Net for Diffusion Models
# -----------------------------------------------------------

class ConditionalUnet1D(nn.Module):
    """
    Conditional 2D U-Net for diffusion models with multi-scale processing.

    This U-Net architecture processes 2D feature maps (e.g., images or masks) and
    supports conditioning on diffusion timesteps and global context. The network
    uses an encoder-decoder structure with skip connections and conditional
    residual blocks for adaptive feature modulation.

    Architecture:
    - Encoder: Progressive downsampling with increasing channels
    - Middle: Deep processing at lowest resolution
    - Decoder: Progressive upsampling with skip connections
    - Output: Projection back to input channel dimension

    Despite the name "1D", this is actually a 2D U-Net implementation.
    """

    def __init__(self,
                 input_dim: int,
                 local_cond_dim: int = None,
                 global_cond_dim: int = None,
                 diffusion_step_embed_dim: int = 256,
                 down_dims=None,
                 kernel_size: int = 3,
                 n_groups: int = 8,
                 cond_predict_scale: bool = False):
        """
        Initialize the conditional U-Net.

        Args:
            input_dim: Number of input/output channels.
            local_cond_dim: Not used in 2D version (kept for API compatibility).
            global_cond_dim: Dimensionality of global conditioning (optional).
            diffusion_step_embed_dim: Dimensionality of timestep embeddings.
            down_dims: Channel dimensions for downsampling path. Default: [64, 128, 256, 512].
            kernel_size: Size of convolutional kernels.
            n_groups: Number of groups for GroupNorm.
            cond_predict_scale: Whether to use scale-shift conditioning.
        """
        super().__init__()

        if down_dims is None:
            down_dims = [64, 128, 256, 512]

        self.input_dim = input_dim
        self.down_dims = list(down_dims)

        # Encoder channels: [C_in, d0, d1, d2, d3]
        self.enc_channels = [input_dim] + self.down_dims
        # Decoder channels: reverse of down_dims
        self.dec_channels = self.down_dims[::-1]

        # --- Timestep encoder for diffusion conditioning ---
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # Combine timestep and global conditioning
        cond_dim = diffusion_step_embed_dim
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        self.local_cond_encoder = None  # Not used in 2D version

        # --- Middle processing blocks at bottleneck ---
        mid_dim = self.dec_channels[0]  # Deepest encoder dimension
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock2D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock2D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        # --- Encoder (downsampling) path ---
        # Downsample at every level: H/2, W/2 per level
        self.down_modules = nn.ModuleList([])
        num_down = len(self.down_dims)
        for i in range(num_down):
            dim_in = self.enc_channels[i]
            dim_out = self.enc_channels[i + 1]

            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock2D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                ConditionalResidualBlock2D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                Downsample2d(dim_out),   # Always downsample after processing
            ]))

        # --- Decoder (upsampling) path ---
        # Upsample as many times as we downsampled
        self.up_modules = nn.ModuleList([])
        num_up = len(self.dec_channels)
        for i in range(num_up):
            # Input channels: current decoder dim + skip connection from encoder
            inC = self.dec_channels[i]
            # Output channels: next decoder dim, or keep same at final level
            outC = self.dec_channels[i] if i == num_up - 1 else self.dec_channels[i + 1]

            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock2D(
                    in_channels=inC * 2,  # Concatenated with skip connection
                    out_channels=outC,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                ConditionalResidualBlock2D(
                    in_channels=outC,
                    out_channels=outC,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                Upsample2d(outC),   # Always upsample after processing
            ]))

        # --- Final projection to output channels ---
        start_dim = self.down_dims[0]  # Shallowest decoder dimension
        self.final_conv = nn.Sequential(
            Conv2dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv2d(start_dim, input_dim, kernel_size=1),  # 1x1 conv for channel projection
        )

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                local_cond: torch.Tensor = None,
                global_cond: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the conditional U-Net.

        Args:
            sample: Input tensor (B, C, H, W) to be processed.
            timestep: Diffusion timestep(s) for conditioning.
            local_cond: Not used in 2D version (kept for API compatibility).
            global_cond: Global conditioning tensor (optional).

        Returns:
            torch.Tensor: Output tensor with same shape as input (B, C, H, W).
        """
        B = sample.shape[0]

        # --- Process timestep conditioning ---
        # Ensure timestep is a tensor on the correct device
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.dim() == 0:
            timestep = timestep[None].to(sample.device)

        # Expand timestep to batch size
        timestep = timestep.expand(B)

        # Encode timestep using sinusoidal embeddings
        global_feature = self.diffusion_step_encoder(timestep)

        # Concatenate with global conditioning if provided
        if global_cond is not None:
            global_feature = torch.cat((global_feature, global_cond), dim=-1)

        # --- Encoder path: downsampling with feature extraction ---
        x = sample
        skips = []  # Store skip connections for decoder
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, global_feature)  # First conditional block
            x = resnet2(x, global_feature)  # Second conditional block
            skips.append(x)  # Save features for skip connection
            x = downsample(x)  # Spatial downsampling

        # --- Middle processing: deep feature processing at bottleneck ---
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # --- Decoder path: upsampling with skip connections ---
        for resnet1, resnet2, upsample in self.up_modules:
            skip = skips.pop()  # Get corresponding encoder features

            # Safety check: resize if spatial dimensions don't match
            if skip.shape[2:] != x.shape[2:]:
                x = torch.nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )

            # Concatenate with skip connection
            x = torch.cat((x, skip), dim=1)

            # Process concatenated features
            x = resnet1(x, global_feature)  # First conditional block
            x = resnet2(x, global_feature)  # Second conditional block
            x = upsample(x)  # Spatial upsampling

        # --- Final projection to output channels ---
        x = self.final_conv(x)   # (B, C_in, H, W)
        return x
