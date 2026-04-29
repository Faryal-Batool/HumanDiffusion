# Module: Image and start-goal conditioning encoders.

import torch
import torch.nn as nn
import torchvision.models as models

# -----------------------------------------------------------
# 1) RGB IMAGE ENCODER  (DiPPeR / DiPPeST Style)
# -----------------------------------------------------------

# Function: Create an RGB ResNet-18 backbone for image feature extraction.
def resnet18_rgb(pretrained: bool = True):
    """
    Build a standard ResNet-18 backbone that accepts **3-channel RGB** input.

    If pretrained=True, we use ImageNet weights (recommended).
    """
    try:
        # Newer torchvision API
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        # Fallback for older versions
        model = models.resnet18(pretrained=pretrained)
    return model


# Class: RGB image encoder that produces global conditioning features.
class DiPPERImageEncoder(nn.Module):
    """
    Encodes an RGB image (img: [B,3,H,W]) into a global feature vector.

    Default uses an RGB ResNet-18 and then flattens the final
    global pooled representation to a vector of size `out_dim`.
    """
    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self, out_dim: int = 512, pretrained: bool = True):
        super().__init__()

        backbone = resnet18_rgb(pretrained=pretrained)
        # Remove the final FC layer, keep conv+pool -> (B,512,1,1)
        layers = list(backbone.children())[:-1]
        self.encoder = nn.Sequential(*layers)  # final: [B,512,1,1]

        self.out_dim = 512
        self.proj = nn.Identity() if out_dim == 512 else nn.Linear(512, out_dim)

    # Function: Run the module forward pass for training or encoding.
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B,3,H,W) RGB image in [0,1] (normalized outside if needed)
        """
        f = self.encoder(img)            # (B,512,1,1)
        g = f.view(f.size(0), -1)        # (B,512)
        return self.proj(g)              # (B,out_dim)


# -----------------------------------------------------------
# 2) START/GOAL EMBEDDING  (DiPPeR Style)
# -----------------------------------------------------------

# Class: MLP encoder for normalized start or goal coordinates.
class DiPPERStartGoalEncoder(nn.Module):
    """
    Embeds (x, y) 2D coordinates into a feature vector.

    INPUT to forward():
        xy: [B,2] with (x_norm, y_norm) in [0,1]^2

    OUTPUT:
        feat: [B,out_dim]
    """
    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self, in_dim: int = 2, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True)
        )
        self.out_dim = out_dim

    # Function: Run the module forward pass for training or encoding.
    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        xy: [B,2], (x_norm, y_norm) in [0,1]^2
        """
        return self.mlp(xy)


# -----------------------------------------------------------
# 3) FULL DiPPeR CONDITIONING MODULE (RGB + START + GOAL)
# -----------------------------------------------------------

# Class: Combined RGB and endpoint conditioning module.
class DiPPERConditioning(nn.Module):
    """
    Combines:
        - RGB image embedding             from img:      [B,3,H,W]
        - Start embedding                 from start_px: [B,2] pixel (x_px, y_px)
        - Goal embedding                  from goal_px:  [B,2] pixel (x_px, y_px)

    GLOBAL DESIGN (PIXEL-SPACE):

      - Dataset & model pass start/end as **pixel coordinates** (x_px, y_px).
      - This module internally normalizes them to [0,1]^2 for the MLP,
        using the spatial size of the image.

    Typical usage in Diffusion:
        cond_vec = DiPPERConditioning(img, start_px, end_px)
        eps_pred = unet(sample=x_t, timestep=t, global_cond=cond_vec)
    """
    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self,
                 img_dim: int = 512,
                 sg_dim: int = 128,
                 final_dim: int = 512,
                 pretrained_backbone: bool = True):
        super().__init__()
        # Encodes RGB image (B,3,H,W) -> (B,img_dim)
        self.img_enc = DiPPERImageEncoder(out_dim=img_dim,
                                          pretrained=pretrained_backbone)
        # Embeds normalized (x_norm,y_norm) -> (B,sg_dim)
        self.sg_enc  = DiPPERStartGoalEncoder(in_dim=2, out_dim=sg_dim)
        # Final fusion to a single conditioning vector
        self.final_fc = nn.Linear(img_dim + 2 * sg_dim, final_dim)

    # Function: Normalize pixel coordinates using the spatial size of the image tensor.
    def _pixels_to_norm(self,
                        xy_px: torch.Tensor,
                        img: torch.Tensor) -> torch.Tensor:
        """
        Convert pixel coordinates (x_px, y_px) to normalized coords (x_norm, y_norm)
        using the spatial size of 'img'.

        INPUT:
          xy_px: (B,2) with (x_px, y_px) in pixel indices
          img  : (B,C,H,W) to infer (H,W)

        OUTPUT:
          xy_norm: (B,2) with values in approximately [0,1].

        NOTE:
          - x_px is divided by (W - 1)
          - y_px is divided by (H - 1)
          - We clamp to [0,1] to be safe.
        """
        assert xy_px.dim() == 2 and xy_px.size(1) == 2, \
            f"DiPPERConditioning._pixels_to_norm: expected (B,2), got {tuple(xy_px.shape)}"
        B, _, H, W = img.shape

        if xy_px.size(0) != B:
            raise ValueError(
                f"DiPPERConditioning: batch size mismatch between coords ({xy_px.size(0)}) "
                f"and img ({B})"
            )

        device = xy_px.device
        dtype = xy_px.dtype

        scale = torch.tensor([W - 1, H - 1], device=device, dtype=dtype)
        xy_norm = xy_px / scale  # broadcast divide
        xy_norm = torch.clamp(xy_norm, 0.0, 1.0)
        return xy_norm

    # Function: Run the module forward pass for training or encoding.
    def forward(self,
                img: torch.Tensor,
                start_px: torch.Tensor,
                goal_px: torch.Tensor) -> torch.Tensor:
        """
        img     : [B,3,H,W] RGB image
        start_px: [B,2] (x_px, y_px) pixel coordinates
        goal_px : [B,2] (x_px, y_px) pixel coordinates

        Returns:
            cond: [B,final_dim] conditioning vector for the UNet.
        """
        # 1) Global RGB feature
        img_feat = self.img_enc(img)  # [B,img_dim]

        # 2) Convert pixel coords -> normalized [0,1]^2 for MLP
        start_norm = self._pixels_to_norm(start_px, img)  # [B,2]
        goal_norm  = self._pixels_to_norm(goal_px, img)   # [B,2]

        # 3) Encode start / goal normalized coords
        start_feat = self.sg_enc(start_norm)  # [B,sg_dim]
        goal_feat  = self.sg_enc(goal_norm)   # [B,sg_dim]

        # 4) Concatenate all and project to final_dim
        cond = torch.cat([img_feat, start_feat, goal_feat], dim=-1)
        return self.final_fc(cond)           # [B,final_dim]
