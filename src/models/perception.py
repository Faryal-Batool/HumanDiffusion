"""
Perception modules for trajectory generation.

This module provides various perception encoders for processing different types of
input data in trajectory generation tasks:

1. LidarImageModel: CNN encoder for processing lidar range images into feature vectors
2. Perception: Main perception module combining lidar, velocity, and target inputs
3. DiPPERImageEncoder: ResNet-18 based RGB image encoder for visual conditioning
4. DiPPERStartGoalEncoder: MLP encoder for start/goal coordinate embeddings
5. DiPPERConditioning: Complete conditioning module combining visual and coordinate features

These modules are used in both CVAE-based and diffusion-based trajectory generation
pipelines to extract relevant features from sensor inputs and conditioning signals.
"""

import time
import warnings
import torch
from torch import nn
from src.utils.configs import DataDict
import torch.nn as nn
import torchvision.models as models


class Perception(nn.Module):
    """
    Main perception module for CVAE-based trajectory generation.

    This module combines multiple sensory inputs into a unified perception encoding:
    - Lidar range images (processed through LidarImageModel)
    - Velocity information (processed through MLP)
    - Target/goal coordinates (passed through directly)

    The combined features are further processed through fusion layers to create
    a rich representation for trajectory generation.

    Used in: CVAE-based trajectory generation pipeline
    """

    def __init__(self, cfg):
        """
        Initialize the perception module.

        Args:
            cfg: Configuration object containing dimensions and hyperparameters
        """
        super(Perception, self).__init__()
        self.cfg = cfg

        # Lidar processing: range image -> feature vector
        self.lidar_model = LidarImageModel(
            input_channel=self.cfg.lidar_num,
            lidar_out_dim=self.cfg.lidar_out,
            norm_layer=self.cfg.lidar_norm_layer
        )

        # Velocity processing: velocity vector -> feature vector
        self.vel_model = nn.Sequential(
            nn.Linear(self.cfg.vel_dim, 64), nn.ELU(),
            nn.Linear(64, 128), nn.ELU(),
            nn.Linear(128, self.cfg.vel_out), nn.LeakyReLU(0.2)
        )

        # Feature fusion: combine all inputs into final perception vector
        # Input dimensions: lidar_features + vel_features + target_coords (2D)
        combo_input_dim = self.cfg.vel_out + self.cfg.lidar_out + 2
        self.combo_layers = nn.Sequential(
            nn.Linear(combo_input_dim, 2 * combo_input_dim), nn.ELU(),
            nn.Linear(2 * combo_input_dim, 2 * combo_input_dim), nn.ELU(),
            nn.Linear(2 * combo_input_dim, combo_input_dim), nn.LeakyReLU(0.2)
        )

    def forward(self, lidar, vel, target):
        """
        Process multi-modal inputs into unified perception features.

        Args:
            lidar: Lidar range image (B, C, H, W)
            vel: Velocity information (B, N, D) - flattened internally
            target: Target/goal coordinates (B, 2)

        Returns:
            perception: Fused feature vector for trajectory generation (B, combo_input_dim)
        """
        # Process lidar range image through CNN
        lidar_fts = self.lidar_model(lidar)  # (B, lidar_out_dim)

        # Process velocity through MLP (flatten temporal/spatial dims first)
        VB, VN, VD = vel.size()
        vel_fts = self.vel_model(vel.view(VB, -1))  # (B, vel_out_dim)

        # Concatenate all features: lidar + velocity + target coordinates
        observation = torch.concat((lidar_fts, vel_fts, target), dim=1)  # (B, lidar_out + vel_out + 2)

        # Final fusion through combination layers
        perception = self.combo_layers(observation)
        return perception
    


# -----------------------------------------------------------
# 1) RGB IMAGE ENCODER  (DiPPeR / DiPPeST Style)
# -----------------------------------------------------------

def resnet18_rgb(pretrained: bool = True):
    """
    Create a ResNet-18 model configured for RGB input.

    This function handles different torchvision versions gracefully,
    using the appropriate API for loading pretrained weights.

    Args:
        pretrained: Whether to load ImageNet pretrained weights

    Returns:
        ResNet-18 model ready for RGB input (3 channels)
    """
    try:
        # Modern torchvision API (v0.13+)
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        # Fallback for older torchvision versions
        model = models.resnet18(pretrained=pretrained)
    return model


class DiPPERImageEncoder(nn.Module):
    """
    RGB image encoder using ResNet-18 backbone.

    This module takes an RGB image and extracts a global feature representation
    using a pretrained ResNet-18 network. The final classification layer is removed,
    and the global average pooled features are projected to the desired output dimension.

    Used in: Diffusion-based trajectory generation for visual conditioning
    """

    def __init__(self, out_dim: int = 512, pretrained: bool = True):
        """
        Initialize the RGB image encoder.

        Args:
            out_dim: Output feature dimension
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()

        # Load ResNet-18 backbone and remove final FC layer
        backbone = resnet18_rgb(pretrained=pretrained)
        # Keep all layers except the final classification layer
        layers = list(backbone.children())[:-1]  # Results in (B, 512, 1, 1)
        self.encoder = nn.Sequential(*layers)

        # ResNet-18 final feature dimension is always 512
        self.out_dim = 512
        # Optional projection to different output dimension
        self.proj = nn.Identity() if out_dim == 512 else nn.Linear(512, out_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image to feature vector.

        Args:
            img: RGB image tensor (B, 3, H, W) in [0,1] range

        Returns:
            Feature vector (B, out_dim)
        """
        # Extract features through ResNet-18
        f = self.encoder(img)            # (B, 512, 1, 1)
        # Flatten spatial dimensions
        g = f.view(f.size(0), -1)        # (B, 512)
        # Optional projection to target dimension
        return self.proj(g)              # (B, out_dim)


# -----------------------------------------------------------
# 2) START/GOAL EMBEDDING  (DiPPeR Style)
# -----------------------------------------------------------

class DiPPERStartGoalEncoder(nn.Module):
    """
    MLP encoder for start/goal coordinate embeddings.

    This module takes normalized 2D coordinates (x_norm, y_norm) in [0,1]^2
    and encodes them into a higher-dimensional feature space through a simple MLP.

    Used for: Encoding start and goal positions in diffusion conditioning
    """

    def __init__(self, in_dim: int = 2, out_dim: int = 128):
        """
        Initialize the coordinate encoder.

        Args:
            in_dim: Input coordinate dimension (default 2 for x,y)
            out_dim: Output embedding dimension
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True)
        )
        self.out_dim = out_dim

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Encode normalized coordinates to feature vector.

        Args:
            xy: Normalized coordinates (B, 2) with values in [0,1]

        Returns:
            Feature embedding (B, out_dim)
        """
        return self.mlp(xy)


# -----------------------------------------------------------
# 3) FULL DiPPeR CONDITIONING MODULE (RGB + START + GOAL)
# -----------------------------------------------------------

class DiPPERConditioning(nn.Module):
    """
    Complete conditioning module for diffusion-based trajectory generation.

    This module combines multiple conditioning signals into a unified feature vector:
    - RGB image features (from DiPPERImageEncoder)
    - Start position embedding (from DiPPERStartGoalEncoder)
    - Goal position embedding (from DiPPERStartGoalEncoder)

    The module handles coordinate normalization internally, converting pixel coordinates
    to normalized [0,1] space for consistent processing.

    Used in: Diffusion model conditioning for trajectory generation
    """

    def __init__(self,
                 img_dim: int = 512,
                 sg_dim: int = 128,
                 final_dim: int = 512,
                 pretrained_backbone: bool = True):
        """
        Initialize the conditioning module.

        Args:
            img_dim: Dimension of RGB image features
            sg_dim: Dimension of start/goal coordinate embeddings
            final_dim: Final output conditioning dimension
            pretrained_backbone: Whether to use pretrained ResNet weights
        """
        super().__init__()

        # RGB image encoder
        self.img_enc = DiPPERImageEncoder(out_dim=img_dim,
                                          pretrained=pretrained_backbone)

        # Start/goal coordinate encoders (shared architecture)
        self.sg_enc = DiPPERStartGoalEncoder(in_dim=2, out_dim=sg_dim)

        # Final fusion layer: combine image + start + goal features
        self.final_fc = nn.Linear(img_dim + 2 * sg_dim, final_dim)

    def _pixels_to_norm(self,
                        xy_px: torch.Tensor,
                        img: torch.Tensor) -> torch.Tensor:
        """
        Convert pixel coordinates to normalized coordinates.

        Pixel coordinates are absolute positions in the image coordinate system,
        while normalized coordinates are in [0,1] relative to image dimensions.
        This normalization ensures consistent processing regardless of image size.

        Args:
            xy_px: Pixel coordinates (B, 2) with (x_px, y_px)
            img: Reference image tensor (B, C, H, W) for dimension info

        Returns:
            Normalized coordinates (B, 2) in [0,1] range
        """
        assert xy_px.dim() == 2 and xy_px.size(1) == 2, \
            f"Expected (B,2) coordinates, got {tuple(xy_px.shape)}"

        B, _, H, W = img.shape

        # Validate batch size consistency
        if xy_px.size(0) != B:
            raise ValueError(
                f"Batch size mismatch: coordinates ({xy_px.size(0)}) vs image ({B})"
            )

        device = xy_px.device
        dtype = xy_px.dtype

        # Normalization scale: divide by (dimension - 1) for proper [0,1] mapping
        scale = torch.tensor([W - 1, H - 1], device=device, dtype=dtype)
        xy_norm = xy_px / scale  # Broadcasting division
        xy_norm = torch.clamp(xy_norm, 0.0, 1.0)  # Ensure bounds
        return xy_norm

    def forward(self,
                img: torch.Tensor,
                start_px: torch.Tensor,
                goal_px: torch.Tensor) -> torch.Tensor:
        """
        Create conditioning vector from RGB image and start/goal positions.

        Args:
            img: RGB image (B, 3, H, W)
            start_px: Start position in pixel coordinates (B, 2)
            goal_px: Goal position in pixel coordinates (B, 2)

        Returns:
            Conditioning vector (B, final_dim) for diffusion model
        """
        # 1) Extract global features from RGB image
        img_feat = self.img_enc(img)  # (B, img_dim)

        # 2) Convert pixel coordinates to normalized [0,1] space
        start_norm = self._pixels_to_norm(start_px, img)  # (B, 2)
        goal_norm = self._pixels_to_norm(goal_px, img)    # (B, 2)

        # 3) Encode normalized coordinates to feature space
        start_feat = self.sg_enc(start_norm)  # (B, sg_dim)
        goal_feat = self.sg_enc(goal_norm)    # (B, sg_dim)

        # 4) Concatenate all features and project to final conditioning dimension
        cond = torch.cat([img_feat, start_feat, goal_feat], dim=-1)
        return self.final_fc(cond)  # (B, final_dim)
