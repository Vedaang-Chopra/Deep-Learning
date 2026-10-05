"""
segmentation_unet.py — U-Net from Scratch + Segmentation Metrics
==================================================================

Student implements:
  - DownBlock: encoder block (conv + pool)
  - UpBlock: decoder block (upsample + skip concat + conv)
  - UNet: full encoder-decoder with skip connections
  - DiceLoss: soft Dice loss for segmentation
  - compute_miou(): mean Intersection-over-Union metric

No prebuilt U-Net allowed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ─────────────────────────────────────────────────────
# Encoder Block
# ─────────────────────────────────────────────────────

class DownBlock(nn.Module):
    """
    U-Net encoder block: two 3x3 convolutions + MaxPool for downsampling.

    Architecture:
      Conv3x3(in_ch, out_ch) → BN → ReLU →
      Conv3x3(out_ch, out_ch) → BN → ReLU →
      MaxPool2d(2)

    Returns BOTH the feature map (before pool) and the pooled output.
    The pre-pool features are needed for skip connections.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  forward(x) must return (features, pooled):           ║
    ║    features = conv_block(x)   ← for skip connection   ║
    ║    pooled = maxpool(features) ← passed to next stage  ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        raise NotImplementedError("TODO: implement DownBlock.__init__()")

    def forward(self, x: torch.Tensor):
        """Returns (features_before_pool, pooled_output)."""
        raise NotImplementedError("TODO: implement DownBlock.forward()")


# ─────────────────────────────────────────────────────
# Decoder Block
# ─────────────────────────────────────────────────────

class UpBlock(nn.Module):
    """
    U-Net decoder block: upsample + skip concat + two 3x3 convolutions.

    Architecture:
      Upsample(x, scale=2) OR ConvTranspose2d →
      Concatenate with skip features along channel dim →
      Conv3x3(in_ch + skip_ch, out_ch) → BN → ReLU →
      Conv3x3(out_ch, out_ch) → BN → ReLU

    Parameters
    ----------
    in_channels : int
        Channels from the lower decoder level.
    skip_channels : int
        Channels from the corresponding encoder skip connection.
    out_channels : int

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  forward(x, skip):                                    ║
    ║  1. Upsample x by 2x (bilinear or ConvTranspose2d)   ║
    ║  2. Crop/pad skip if spatial dims don't match exactly ║
    ║  3. Concatenate [x, skip] along channel dim           ║
    ║  4. Pass through conv block                            ║
    ║                                                       ║
    ║  Hint: Use F.interpolate(x, size=skip.shape[2:])      ║
    ║  for safe spatial matching.                            ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        raise NotImplementedError("TODO: implement UpBlock.__init__()")

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: implement UpBlock.forward()")


# ─────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.

    Architecture (default channels=[64, 128, 256, 512]):

      Encoder:
        DownBlock(in, 64) → DownBlock(64, 128) → DownBlock(128, 256) → DownBlock(256, 512)

      Bottleneck:
        Conv3x3(512, 1024) → BN → ReLU → Conv3x3(1024, 1024) → BN → ReLU

      Decoder:
        UpBlock(1024, 512, 512) → UpBlock(512, 256, 256) → UpBlock(256, 128, 128) → UpBlock(128, 64, 64)

      Head:
        Conv1x1(64, num_classes)

    Parameters
    ----------
    in_channels : int (default 3, for RGB images)
    num_classes : int (default 21, Pascal VOC)
    base_channels : int (default 64)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  __init__:                                            ║
    ║  1. Build encoder as nn.ModuleList of DownBlocks      ║
    ║  2. Build bottleneck (two conv layers)                 ║
    ║  3. Build decoder as nn.ModuleList of UpBlocks        ║
    ║  4. Final 1x1 conv for class logits                   ║
    ║                                                       ║
    ║  forward(x):                                          ║
    ║  1. Pass through encoder, collecting skip features     ║
    ║  2. Pass through bottleneck                            ║
    ║  3. Pass through decoder with skip connections         ║
    ║     (reverse order!)                                   ║
    ║  4. Final 1x1 conv → return logits (B, C, H, W)       ║
    ║                                                       ║
    ║  Hint: Store encoder features in a list, then          ║
    ║  pop() or reverse-iterate during decoding.             ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        base_channels: int = 64,
    ):
        super().__init__()
        raise NotImplementedError("TODO: implement UNet.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (B, num_classes, H, W)."""
        raise NotImplementedError("TODO: implement UNet.forward()")


# ─────────────────────────────────────────────────────
# Dice Loss
# ─────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice

    For multi-class, compute per-class Dice and average.

    Parameters
    ----------
    smooth : float (default 1.0)
        Smoothing factor to avoid division by zero.
    ignore_index : int (default -100)
        Class index to ignore.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement forward.                             ║
    ║                                                       ║
    ║  forward(logits, targets):                            ║
    ║  1. Convert logits to probabilities (softmax dim=1)   ║
    ║  2. One-hot encode targets: (B,H,W) → (B,C,H,W)     ║
    ║  3. For each class c:                                  ║
    ║     intersection = sum(probs[:,c] * onehot[:,c])       ║
    ║     dice_c = (2*intersection + smooth) /               ║
    ║              (sum(probs[:,c]) + sum(onehot[:,c]) +      ║
    ║               smooth)                                  ║
    ║  4. Loss = 1 - mean(dice_c)                            ║
    ║                                                       ║
    ║  Hint: Use F.one_hot(targets, num_classes).permute()   ║
    ║  to get (B, C, H, W) format.                          ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: implement DiceLoss.forward()")


# ─────────────────────────────────────────────────────
# mIoU Metric
# ─────────────────────────────────────────────────────

def compute_miou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
) -> dict:
    """
    Compute mean Intersection-over-Union (mIoU).

    Parameters
    ----------
    preds : (B, H, W) — predicted class indices
    targets : (B, H, W) — ground truth class indices
    num_classes : int
    ignore_index : int

    Returns
    -------
    dict with:
      'miou': float — mean IoU across classes present
      'per_class_iou': dict mapping class_id → IoU

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  For each class c:                                     ║
    ║    intersection = ((preds == c) & (targets == c)).sum()║
    ║    union = ((preds == c) | (targets == c)).sum()       ║
    ║    iou_c = intersection / (union + 1e-8)               ║
    ║  Skip classes not present in targets.                  ║
    ║  miou = mean of per-class IoUs.                        ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement compute_miou()")


# ─────────────────────────────────────────────────────
# Synthetic Segmentation Dataset (provided)
# ─────────────────────────────────────────────────────

class SyntheticShapesDataset(torch.utils.data.Dataset):
    """
    A synthetic segmentation dataset with colored shapes on black background.

    Classes: 0=background, 1=circle, 2=rectangle, 3=triangle
    Images: (3, H, W) RGB
    Masks: (H, W) class indices

    This is provided so the student can focus on U-Net implementation.
    """

    def __init__(self, num_samples: int = 500, img_size: int = 128, num_classes: int = 4):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.images, self.masks = self._generate()

    def _generate(self):
        import random
        images, masks = [], []
        H = W = self.img_size
        for _ in range(self.num_samples):
            img = torch.zeros(3, H, W)
            mask = torch.zeros(H, W, dtype=torch.long)
            n_shapes = random.randint(1, 4)
            for _ in range(n_shapes):
                shape_type = random.randint(1, 3)
                color = torch.rand(3)
                cx, cy = random.randint(15, H - 15), random.randint(15, W - 15)
                r = random.randint(8, 20)
                yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
                if shape_type == 1:  # circle
                    region = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
                elif shape_type == 2:  # rectangle
                    region = (xx >= cx - r) & (xx < cx + r) & (yy >= cy - r) & (yy < cy + r)
                else:  # triangle (simplified)
                    region = ((yy - (cy - r)) >= 0) & ((yy - (cy - r)) <= 2 * r) & \
                             (torch.abs(xx - cx) <= (r - (yy - (cy - r)) * r / (2 * r + 1e-5)))
                img[:, region] = color.unsqueeze(1)
                mask[region] = shape_type
            images.append(img)
            masks.append(mask)
        return images, masks

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]
