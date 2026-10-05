"""
yolo_v1.py — YOLOv1 Core Logic from First Principles
======================================================

Student implements:
  - YOLOv1Head: prediction head (grid → boxes + confidence + classes)
  - YOLOv1Loss: multi-part loss function
  - decode_predictions(): convert grid predictions to bounding boxes

No high-level detection libraries allowed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


# ─────────────────────────────────────────────────────
# YOLOv1 Head
# ─────────────────────────────────────────────────────

class YOLOv1Head(nn.Module):
    """
    YOLOv1 prediction head.

    Takes a feature map from a CNN backbone and predicts, for each grid cell:
      - B bounding boxes, each with (x, y, w, h, confidence) = 5 values
      - C class probabilities (shared across all B boxes in the cell)

    Output shape: (batch, S, S, B*5 + C)

    Parameters
    ----------
    backbone_channels : int
        Number of channels from the CNN backbone feature map.
    S : int (default 7)
        Grid size (S x S).
    B : int (default 2)
        Number of bounding boxes per cell.
    C : int (default 10)
        Number of classes.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__ and forward.                ║
    ║                                                       ║
    ║  Architecture suggestion:                             ║
    ║  1. AdaptiveAvgPool2d(S) → force spatial to SxS       ║
    ║  2. Conv2d to reduce channels                         ║
    ║  3. Final Conv2d → (B*5 + C) channels                 ║
    ║  4. Reshape to (batch, S, S, B*5 + C)                 ║
    ║                                                       ║
    ║  Alternatively, flatten + FC layers:                   ║
    ║  1. AdaptiveAvgPool2d(S) → flatten                    ║
    ║  2. Linear → ReLU → Dropout                           ║
    ║  3. Linear → S * S * (B*5 + C)                        ║
    ║  4. Reshape                                            ║
    ║                                                       ║
    ║  The output is raw (no activation on coords/conf yet). ║
    ║  Activations are applied in the loss function.         ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        backbone_channels: int = 256,
        S: int = 7,
        B: int = 2,
        C: int = 10,
    ):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        raise NotImplementedError("TODO: implement YOLOv1Head.__init__()")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (batch, backbone_channels, H, W) from CNN backbone

        Returns
        -------
        (batch, S, S, B*5 + C) prediction tensor
        """
        raise NotImplementedError("TODO: implement YOLOv1Head.forward()")


# ─────────────────────────────────────────────────────
# YOLOv1 Loss
# ─────────────────────────────────────────────────────

class YOLOv1Loss(nn.Module):
    """
    YOLOv1 multi-part loss function.

    The loss combines four components:

    1. **Coordinate loss** (λ_coord = 5.0):
       MSE of (x, y) center coordinates for responsible boxes
       MSE of (√w, √h) — square root for scale-invariance

    2. **Object confidence loss** (λ_obj = 1.0):
       MSE of predicted confidence vs IoU with ground truth
       (for cells that DO contain an object)

    3. **No-object confidence loss** (λ_noobj = 0.5):
       MSE of predicted confidence vs 0
       (for cells that do NOT contain an object)

    4. **Classification loss** (λ_class = 1.0):
       MSE of predicted class probs vs one-hot target
       (for cells that contain an object)

    Parameters
    ----------
    S : int (default 7)
    B : int (default 2)
    C : int (default 10)
    lambda_coord : float (default 5.0)
    lambda_noobj : float (default 0.5)

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement forward.                             ║
    ║                                                       ║
    ║  forward(predictions, targets):                       ║
    ║    predictions: (batch, S, S, B*5 + C)                ║
    ║    targets:     (batch, S, S, 5 + C)                  ║
    ║      target format per cell:                          ║
    ║        [x, y, w, h, objectness, class_0, ..., class_C]║
    ║      objectness = 1 if object center is in cell       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Extract object mask: cells where target obj == 1   ║
    ║  2. For cells WITH object:                             ║
    ║     a. Extract B predicted boxes                        ║
    ║     b. Compute IoU of each with GT box                 ║
    ║     c. "Responsible" box = highest IoU                  ║
    ║     d. Coordinate loss on responsible box               ║
    ║     e. Confidence loss: target = IoU value              ║
    ║     f. Classification loss on class probs               ║
    ║  3. For cells WITHOUT object:                          ║
    ║     a. No-object confidence loss for ALL B boxes       ║
    ║  4. Total = λ_coord*coord + obj + λ_noobj*noobj +      ║
    ║            class_loss                                   ║
    ║  5. Return total AND individual components (for logging)║
    ║                                                       ║
    ║  Hints:                                               ║
    ║  - Use compute_iou from detection_metrics.py          ║
    ║  - Apply sigmoid to confidence predictions             ║
    ║  - Apply sigmoid to x, y center predictions            ║
    ║  - Take sqrt of w, h before MSE (and of targets too)   ║
    ║  - Watch dimensions carefully!                         ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        S: int = 7,
        B: int = 2,
        C: int = 10,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    ):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns
        -------
        total_loss : scalar tensor
        components : dict with keys 'coord', 'obj', 'noobj', 'class'
        """
        raise NotImplementedError("TODO: implement YOLOv1Loss.forward()")


# ─────────────────────────────────────────────────────
# Decode Predictions
# ─────────────────────────────────────────────────────

def decode_predictions(
    predictions: torch.Tensor,
    S: int = 7,
    B: int = 2,
    C: int = 10,
    conf_threshold: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert YOLO grid predictions to bounding boxes in image coordinates.

    Parameters
    ----------
    predictions : (S, S, B*5 + C) for a single image
    S, B, C : grid/box/class parameters
    conf_threshold : minimum confidence to keep a box

    Returns
    -------
    boxes : (K, 4) in xyxy format, normalized [0, 1]
    scores : (K,) confidence * class_prob
    class_ids : (K,) predicted class indices

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  For each cell (i, j) and each box b:                  ║
    ║  1. Extract (x, y, w, h, conf) for box b              ║
    ║  2. Convert cell-relative coords to image-relative:   ║
    ║     x_img = (j + sigmoid(x)) / S                       ║
    ║     y_img = (i + sigmoid(y)) / S                       ║
    ║     w_img = w (already relative to image)              ║
    ║     h_img = h                                          ║
    ║  3. Convert (cx, cy, w, h) → (x1, y1, x2, y2)        ║
    ║  4. Compute score = sigmoid(conf) * max(class_probs)   ║
    ║  5. Keep boxes with score > conf_threshold              ║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement decode_predictions()")


# ─────────────────────────────────────────────────────
# Simple Detection Dataset (provided)
# ─────────────────────────────────────────────────────

class SimpleDetectionDataset(torch.utils.data.Dataset):
    """
    A synthetic dataset for YOLO training with simple shapes.

    Each image has 1-3 randomly placed colored shapes on a dark background.
    Labels are in YOLO grid format: (S, S, 5 + C).

    Provided utility — no student implementation needed.
    """

    def __init__(
        self,
        num_samples: int = 500,
        img_size: int = 224,
        S: int = 7,
        C: int = 3,
    ):
        import random
        self.num_samples = num_samples
        self.img_size = img_size
        self.S = S
        self.C = C
        self.images = []
        self.targets = []

        for _ in range(num_samples):
            img = torch.rand(3, img_size, img_size) * 0.1  # dark background
            target = torch.zeros(S, S, 5 + C)
            n_objects = random.randint(1, 3)

            for _ in range(n_objects):
                cls = random.randint(0, C - 1)
                cx = random.uniform(0.15, 0.85)
                cy = random.uniform(0.15, 0.85)
                w = random.uniform(0.08, 0.25)
                h = random.uniform(0.08, 0.25)

                # Draw shape on image
                x1_px = int((cx - w / 2) * img_size)
                y1_px = int((cy - h / 2) * img_size)
                x2_px = int((cx + w / 2) * img_size)
                y2_px = int((cy + h / 2) * img_size)
                x1_px, y1_px = max(0, x1_px), max(0, y1_px)
                x2_px, y2_px = min(img_size, x2_px), min(img_size, y2_px)
                color = torch.zeros(3)
                color[cls % 3] = 0.6 + random.random() * 0.4
                img[:, y1_px:y2_px, x1_px:x2_px] = color.view(3, 1, 1)

                # Assign to grid cell
                gi = int(cy * S)
                gj = int(cx * S)
                gi = min(gi, S - 1)
                gj = min(gj, S - 1)
                if target[gi, gj, 4] == 0:  # cell not yet assigned
                    x_cell = cx * S - gj
                    y_cell = cy * S - gi
                    target[gi, gj, 0] = x_cell
                    target[gi, gj, 1] = y_cell
                    target[gi, gj, 2] = w
                    target[gi, gj, 3] = h
                    target[gi, gj, 4] = 1.0  # objectness
                    target[gi, gj, 5 + cls] = 1.0

            self.images.append(img)
            self.targets.append(target)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]
