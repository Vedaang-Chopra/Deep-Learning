#!/usr/bin/env python3
"""Generate the Group 3 Vision Systems Lab notebook."""
import json, os

def md(source):
    if isinstance(source, str): source = source.split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in source[:-1]] + [source[-1]]}

def code(source):
    if isinstance(source, str): source = source.split("\n")
    return {"cell_type": "code", "metadata": {}, "source": [l + "\n" for l in source[:-1]] + [source[-1]], "execution_count": None, "outputs": []}

cells = []

# ═══════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════
cells.append(md("""\
# 👁️ Notebook 3 — Vision Systems Lab
## CNN Internals, Segmentation & Detection Mechanics

**Group 3 — CNNs & Vision Systems**

---

### 🎯 Learning Objectives

1. Build CNN architectures from composable blocks (ConvBlock, ResidualBlock)
2. Understand receptive fields, feature hierarchies, and residual connections
3. Implement transfer learning with manual backbone freezing
4. Build U-Net from scratch for semantic segmentation
5. Implement IoU, NMS, and mAP from first principles
6. Implement YOLOv1 head and multi-part loss
7. Debug vision system failures systematically

### 📂 File Structure

```
Group_3_Vision_Systems/
├── notebooks/
│   └── 03_vision_systems_lab.ipynb   ← you are here
└── src/
    ├── cnn_blocks.py            ← ConvBlock, ResidualBlock, PlainCNN, ResidualCNN
    ├── segmentation_unet.py     ← UNet, DiceLoss, mIoU, synthetic dataset
    ├── detection_metrics.py     ← IoU, NMS, mAP (no torchvision.ops!)
    └── yolo_v1.py               ← YOLOv1Head, YOLOv1Loss, decode, synthetic dataset
```

> ⚠️ **No shortcuts**: You must implement IoU, NMS, and mAP manually. `torchvision.ops` is banned."""))

# ═══════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════
cells.append(md("## 0 — Environment Setup"))

cells.append(code("""\
import sys, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.pardir, "src"))

from cnn_blocks import ConvBlock, ResidualBlock, PlainCNN, ResidualCNN, compute_receptive_field
from segmentation_unet import DownBlock, UpBlock, UNet, DiceLoss, compute_miou, SyntheticShapesDataset
from detection_metrics import (
    compute_iou, non_max_suppression, compute_ap, mean_average_precision,
    visualize_boxes, visualize_nms_comparison,
)
from yolo_v1 import YOLOv1Head, YOLOv1Loss, decode_predictions, SimpleDetectionDataset

print(f"PyTorch version : {torch.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device          : {DEVICE}")"""))

cells.append(code("""\
# ── CIFAR-10 for classification experiments ──
train_transform = T.Compose([
    T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
    T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
test_transform = T.Compose([
    T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
cifar_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

# Use subset for speed
SUBSET = 5000
cifar_train_sub = torch.utils.data.Subset(cifar_train, range(SUBSET))
print(f"CIFAR-10: {SUBSET} train / {len(cifar_test)} test samples")"""))

# ═══════════════════════════════════════════════════════
# SECTION 1 — CNN ARCHITECTURE ENGINEERING
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 1 — CNN Architecture Engineering

## 1.1 Conceptual Background

### Convolution Math

A 2D convolution slides a kernel over the input feature map:

```
Output size = floor((Input + 2*Padding - Kernel) / Stride) + 1
```

### Receptive Field

The **receptive field** is the region of the input image that influences a single output neuron.

```
Layer 1: 3×3 conv → RF = 3
Layer 2: 3×3 conv → RF = 5
Layer 3: 3×3 conv → RF = 7
```

Two stacked 3×3 convs have the same RF as one 5×5, but with **fewer parameters** and **more nonlinearity**. This is the VGG insight.

### Why Residual Connections?

In plain deep networks:
- Gradient signal degrades through many layers (vanishing)
- Optimization landscape becomes harder (saddle points)
- Adding more layers can HURT performance

**Residual connections** let gradients flow directly through the identity path:

```
            ┌──────────────────────┐
            │                      │
    x ──→ ConvBlock → ConvBlock → (+) → output
    │                              ↑
    └──────── identity ────────────┘
```

The network only needs to learn the **residual** F(x), not the full mapping H(x) = F(x) + x."""))

cells.append(md("""\
## 1.2 Implementation Tasks

Open `src/cnn_blocks.py` and implement:

1. **`ConvBlock`** — Conv2d + optional BatchNorm + activation
2. **`ResidualBlock`** — skip connection with shape alignment
3. **`PlainCNN`** — stacked ConvBlocks (no residuals)
4. **`ResidualCNN`** — stacked ResidualBlocks"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify CNN blocks                      ║
# ╚═══════════════════════════════════════════════════════╝

# Test ConvBlock
cb = ConvBlock(3, 64, kernel_size=3, stride=1, padding=1)
x = torch.randn(2, 3, 32, 32)
out = cb(x)
assert out.shape == (2, 64, 32, 32), f"ConvBlock: expected (2,64,32,32) got {out.shape}"
print(f"  ✅ ConvBlock: {x.shape} → {out.shape}")

# Test ResidualBlock (same channels)
rb = ResidualBlock(64, 64)
x = torch.randn(2, 64, 32, 32)
out = rb(x)
assert out.shape == (2, 64, 32, 32), f"ResidualBlock: expected (2,64,32,32) got {out.shape}"
print(f"  ✅ ResidualBlock (same ch): {x.shape} → {out.shape}")

# Test ResidualBlock (channel change + stride)
rb2 = ResidualBlock(64, 128, stride=2)
out2 = rb2(x)
assert out2.shape == (2, 128, 16, 16), f"ResidualBlock: expected (2,128,16,16) got {out2.shape}"
print(f"  ✅ ResidualBlock (64→128, stride=2): {x.shape} → {out2.shape}")

# Test PlainCNN
plain_model = PlainCNN(num_classes=10)
x = torch.randn(2, 3, 32, 32)
out = plain_model(x)
assert out.shape == (2, 10), f"PlainCNN: expected (2,10) got {out.shape}"
print(f"  ✅ PlainCNN: {x.shape} → {out.shape}  ({sum(p.numel() for p in plain_model.parameters()):,} params)")

# Test ResidualCNN
res_model = ResidualCNN(num_classes=10)
out = res_model(x)
assert out.shape == (2, 10), f"ResidualCNN: expected (2,10) got {out.shape}"
print(f"  ✅ ResidualCNN: {x.shape} → {out.shape}  ({sum(p.numel() for p in res_model.parameters()):,} params)")"""))

cells.append(md("### 1.3 Receptive Field Analysis"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  Compute receptive field for your CNN architecture     ║
# ║  Use the provided compute_receptive_field() utility    ║
# ╚═══════════════════════════════════════════════════════╝

# Example: define your architecture layers
layers = [
    {"name": "conv1", "kernel": 3, "stride": 1, "padding": 1},
    {"name": "conv2", "kernel": 3, "stride": 1, "padding": 1},
    {"name": "pool1", "kernel": 2, "stride": 2, "padding": 0},
    {"name": "conv3", "kernel": 3, "stride": 1, "padding": 1},
    {"name": "conv4", "kernel": 3, "stride": 1, "padding": 1},
    {"name": "pool2", "kernel": 2, "stride": 2, "padding": 0},
]

rf_results = compute_receptive_field(layers)
print(f"{'Layer':<10} {'Kernel':<8} {'Stride':<8} {'RF':<8} {'Jump':<8}")
print("-" * 42)
for r in rf_results:
    print(f"{r['name']:<10} {r['kernel']:<8} {r['stride']:<8} {r['rf']:<8} {r['jump']:<8}")"""))

cells.append(md("### 1.4 Experiment — Plain CNN vs Residual CNN"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Compare PlainCNN vs ResidualCNN           ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Train PlainCNN for 10 epochs on CIFAR-10 subset    ║
# ║  2. Train ResidualCNN for 10 epochs on same data       ║
# ║  3. Plot training loss + val accuracy for both         ║
# ║  4. Compare gradient norms (use Group 2 tools if avail)║
# ║  5. Which converges faster? Which achieves higher acc? ║
# ╚═══════════════════════════════════════════════════════╝

# Minimal training loop
def train_classifier(model, train_dl, test_dl, epochs=10, lr=1e-3):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses, accs = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(train_dl))

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                correct += (model(imgs).argmax(1) == labels).sum().item()
                total += labels.size(0)
        accs.append(100 * correct / total)
        print(f"  Epoch {epoch+1}: loss={losses[-1]:.4f} acc={accs[-1]:.1f}%")
    return {"train_loss": losses, "val_acc": accs}

train_dl = torch.utils.data.DataLoader(cifar_train_sub, batch_size=64, shuffle=True, num_workers=0)
test_dl = torch.utils.data.DataLoader(cifar_test, batch_size=256, shuffle=False, num_workers=0)

# TODO: Train both models and compare
# print("=== PlainCNN ===")
# plain_results = train_classifier(PlainCNN(), train_dl, test_dl)
# print("\\n=== ResidualCNN ===")
# res_results = train_classifier(ResidualCNN(), train_dl, test_dl)"""))

cells.append(md("""\
### 1.5 Reflection

1. **Does the residual connection help gradient flow?** How can you verify this empirically?
2. **What happens if you remove BatchNorm from the CNN?** Try it and observe.
3. **Why do two stacked 3×3 convs beat one 5×5?** What are the tradeoffs?"""))

# ═══════════════════════════════════════════════════════
# SECTION 2 — TRANSFER LEARNING
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 2 — Transfer Learning

## 2.1 Conceptual Background

### Feature Reuse

Early CNN layers learn generic features (edges, textures) that transfer across tasks.
Later layers learn task-specific features.

| Strategy | What | When |
|----------|------|------|
| **Frozen backbone** | Only train the new head | Small dataset, similar domain |
| **Full finetune** | Train everything (low LR) | Larger dataset, different domain |
| **Gradual unfreezing** | Unfreeze layers over time | Best of both worlds |

### Catastrophic Forgetting

If you finetune with a high LR, the pretrained features are destroyed.
**Fix**: Use a much lower LR for pretrained layers than for the new head.

### Practical Implementation

```python
# Freeze: set requires_grad = False
for param in model.backbone.parameters():
    param.requires_grad = False

# Only optimizer sees unfrozen params
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
```"""))

cells.append(md("## 2.2 Implementation Task — Transfer Learning with ResNet"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Transfer Learning on CIFAR-10             ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Load pretrained ResNet-18 (torchvision)            ║
# ║  2. Replace the final FC layer for 10 classes          ║
# ║  3. Experiment A: FREEZE backbone, train head only     ║
# ║     - Set requires_grad=False for all backbone params  ║
# ║     - Only pass head params to optimizer               ║
# ║  4. Experiment B: FINETUNE entire model (lower LR)     ║
# ║  5. Compare: accuracy, training time, convergence      ║
# ║                                                       ║
# ║  Hints:                                               ║
# ║  - model = torchvision.models.resnet18(pretrained=True)║
# ║  - model.fc = nn.Linear(512, 10)                      ║
# ║  - For frozen: optimizer only gets model.fc.parameters()║
# ║  - For finetune: use lower LR (e.g., 1e-4)            ║
# ║  - CIFAR is 32x32 but ResNet expects 224x224.          ║
# ║    Either resize or use a smaller ResNet variant.       ║
# ╚═══════════════════════════════════════════════════════╝

# TODO: Implement and compare frozen vs finetune
"""))

cells.append(md("""\
### 2.3 Reflection

1. **When is freezing better than finetuning?** Give a concrete example.
2. **How would you implement gradual unfreezing?** Sketch the algorithm.
3. **What's the risk of a too-high LR during finetuning?**"""))

# ═══════════════════════════════════════════════════════
# SECTION 3 — SEMANTIC SEGMENTATION (U-NET)
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 3 — Semantic Segmentation: U-Net from Scratch

## 3.1 Conceptual Background

### Encoder-Decoder Architecture

Segmentation requires **dense prediction** — a class label for every pixel. The encoder compresses spatial information, the decoder recovers it.

```
Encoder (↓ resolution, ↑ channels)    Decoder (↑ resolution, ↓ channels)
                                       
Input (3,H,W) → [64,H/2] → [128,H/4] → [256,H/8] → Bottleneck
                    │             │             │
                    └─── skip ────┘             │
                          │       └── skip ─────┘
                          │             │
                    [64,H/2] ← [128,H/4] ← [256,H/8] ← Decoder
                       │
                  Output (C,H,W)
```

### Skip Connections (U-Net's Key Innovation)

The decoder loses fine spatial details. Skip connections concatenate high-resolution encoder features with decoder features, providing both:
- **Semantic info** (from deep layers)
- **Spatial info** (from shallow layers)

### Dice Loss vs Cross-Entropy

| Loss | Pros | Cons |
|------|------|------|
| **Cross-Entropy** | Simple, well-understood | Dominated by background class |
| **Dice Loss** | Handles class imbalance | Can be unstable, less smooth |
| **CE + Dice** | Best of both | Slightly more complex |

### Dice Formula

```
Dice(A, B) = 2|A ∩ B| / (|A| + |B|)
```

Dice = 1 means perfect overlap. Loss = 1 - Dice."""))

cells.append(md("""\
## 3.2 Implementation Tasks

Open `src/segmentation_unet.py` and implement:

1. **`DownBlock`** — encoder block (conv + pool)
2. **`UpBlock`** — decoder block (upsample + skip concat + conv)
3. **`UNet`** — full encoder-decoder
4. **`DiceLoss`** — soft Dice loss
5. **`compute_miou()`** — mean IoU metric"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify U-Net components                 ║
# ╚═══════════════════════════════════════════════════════╝

# Test DownBlock
db = DownBlock(3, 64)
x = torch.randn(2, 3, 128, 128)
features, pooled = db(x)
assert features.shape == (2, 64, 128, 128), f"DownBlock features: {features.shape}"
assert pooled.shape == (2, 64, 64, 64), f"DownBlock pooled: {pooled.shape}"
print(f"  ✅ DownBlock: {x.shape} → features {features.shape}, pooled {pooled.shape}")

# Test UpBlock
ub = UpBlock(128, 64, 64)
x_up = torch.randn(2, 128, 32, 32)
skip = torch.randn(2, 64, 64, 64)
out = ub(x_up, skip)
assert out.shape == (2, 64, 64, 64), f"UpBlock: {out.shape}"
print(f"  ✅ UpBlock: input {x_up.shape} + skip {skip.shape} → {out.shape}")

# Test full U-Net
unet = UNet(in_channels=3, num_classes=4, base_channels=32)
x = torch.randn(2, 3, 128, 128)
out = unet(x)
assert out.shape == (2, 4, 128, 128), f"UNet: expected (2,4,128,128) got {out.shape}"
print(f"  ✅ UNet: {x.shape} → {out.shape}  ({sum(p.numel() for p in unet.parameters()):,} params)")

# Test DiceLoss
dice = DiceLoss()
logits = torch.randn(2, 4, 32, 32)
targets = torch.randint(0, 4, (2, 32, 32))
loss = dice(logits, targets)
assert loss.shape == (), f"DiceLoss should return scalar, got {loss.shape}"
print(f"  ✅ DiceLoss: {loss.item():.4f}")

# Test mIoU
preds = torch.randint(0, 4, (2, 32, 32))
result = compute_miou(preds, targets, num_classes=4)
assert "miou" in result, "compute_miou must return 'miou'"
print(f"  ✅ mIoU: {result['miou']:.4f}")"""))

cells.append(md("### 3.3 Experiment — Train U-Net on Synthetic Shapes"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Train U-Net on SyntheticShapesDataset     ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create SyntheticShapesDataset (500 samples, 128px) ║
# ║  2. Split into train/val                                ║
# ║  3. Build U-Net (4 classes, small base_channels=32)     ║
# ║  4. Train for 15-20 epochs with DiceLoss + CE          ║
# ║  5. Plot: mIoU per epoch, train loss                   ║
# ║  6. Visualize: sample prediction vs ground truth        ║
# ╚═══════════════════════════════════════════════════════╝

seg_dataset = SyntheticShapesDataset(num_samples=500, img_size=128, num_classes=4)
print(f"Segmentation dataset: {len(seg_dataset)} samples")
print(f"  Image shape: {seg_dataset[0][0].shape}")
print(f"  Mask shape:  {seg_dataset[0][1].shape}")
print(f"  Classes:     {seg_dataset[0][1].unique().tolist()}")

# TODO: Split, train, evaluate, plot
"""))

cells.append(md("""\
### 3.4 Experiment — What Happens Without Skip Connections?

Modify your U-Net to disable skip connections (pass zeros instead of encoder features).
Re-train and compare mIoU. This demonstrates why skip connections are essential."""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  TODO: Train U-Net WITHOUT skip connections            ║
# ║        Compare mIoU with the full U-Net                ║
# ║        Visualize predictions to see the difference     ║
# ╚═══════════════════════════════════════════════════════╝
"""))

# ═══════════════════════════════════════════════════════
# SECTION 4 — OBJECT DETECTION FUNDAMENTALS
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 4 — Object Detection Fundamentals

## 4.1 Conceptual Background

### Bounding Box Formats

| Format | Fields | Used By |
|--------|--------|---------|
| **xyxy** | (x1, y1, x2, y2) — corners | COCO, standard |
| **xywh** | (cx, cy, w, h) — center+size | YOLO |

### Intersection over Union (IoU)

```
       ┌──────────┐
       │  Box A   │
       │    ┌─────┼────┐
       │    │ A∩B │    │
       └────┼─────┘    │
            │  Box B   │
            └──────────┘

IoU = Area(A ∩ B) / Area(A ∪ B)
    = Area(A ∩ B) / (Area(A) + Area(B) - Area(A ∩ B))
```

IoU = 1.0 means perfect overlap. IoU = 0 means no overlap.

### Non-Max Suppression (NMS)

Detectors often predict multiple overlapping boxes for the same object.
NMS keeps only the best one:

1. Sort boxes by confidence (descending)
2. Take the highest-confidence box → keep it
3. Remove all boxes with IoU > threshold with the kept box
4. Repeat until no boxes remain

### Mean Average Precision (mAP)

For each class:
1. Rank all predictions by confidence
2. At each prediction, compute cumulative precision and recall
3. AP = area under the precision-recall curve (11-point interpolation)

mAP = mean of AP across all classes."""))

cells.append(md("""\
## 4.2 Implementation Tasks

Open `src/detection_metrics.py` and implement:

1. **`compute_iou(box1, box2)`** — pairwise IoU (no torchvision.ops!)
2. **`non_max_suppression(boxes, scores, threshold)`** — NMS algorithm
3. **`compute_ap(recalls, precisions)`** — 11-point interpolation AP
4. **`mean_average_precision(pred_boxes, gt_boxes)`** — full mAP"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify IoU                             ║
# ╚═══════════════════════════════════════════════════════╝

# Perfect overlap → IoU = 1.0
box_a = torch.tensor([[0, 0, 10, 10]], dtype=torch.float)
box_b = torch.tensor([[0, 0, 10, 10]], dtype=torch.float)
iou = compute_iou(box_a, box_b)
assert abs(iou.item() - 1.0) < 1e-5, f"Perfect overlap IoU should be 1.0, got {iou.item()}"
print(f"  ✅ IoU (perfect overlap): {iou.item():.4f}")

# No overlap → IoU = 0.0
box_c = torch.tensor([[20, 20, 30, 30]], dtype=torch.float)
iou = compute_iou(box_a, box_c)
assert abs(iou.item()) < 1e-5, f"No overlap IoU should be 0.0, got {iou.item()}"
print(f"  ✅ IoU (no overlap): {iou.item():.4f}")

# Partial overlap
box_d = torch.tensor([[5, 5, 15, 15]], dtype=torch.float)
iou = compute_iou(box_a, box_d)
expected_iou = 25.0 / (100 + 100 - 25)
assert abs(iou.item() - expected_iou) < 1e-4, f"Expected {expected_iou:.4f}, got {iou.item():.4f}"
print(f"  ✅ IoU (partial overlap): {iou.item():.4f} (expected {expected_iou:.4f})")

# Batch IoU: (N, M) pairwise
boxes1 = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float)
boxes2 = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [3, 3, 8, 8]], dtype=torch.float)
iou_matrix = compute_iou(boxes1, boxes2)
assert iou_matrix.shape == (2, 3), f"Expected (2,3) got {iou_matrix.shape}"
print(f"  ✅ Batch IoU: {boxes1.shape[0]}×{boxes2.shape[0]} → {iou_matrix.shape}")"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify NMS                             ║
# ╚═══════════════════════════════════════════════════════╝

# Create overlapping boxes around the same object
boxes = torch.tensor([
    [10, 10, 50, 50],   # high confidence
    [12, 12, 52, 52],   # overlapping
    [11, 11, 51, 51],   # overlapping
    [100, 100, 150, 150],  # separate object
], dtype=torch.float)
scores = torch.tensor([0.9, 0.75, 0.8, 0.85])

keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
print(f"  Boxes before NMS: {len(boxes)}")
print(f"  Boxes after NMS:  {len(keep)}")
print(f"  Kept indices: {keep.tolist()}")
assert len(keep) == 2, f"Expected 2 boxes after NMS, got {len(keep)}"
print(f"  ✅ NMS correctly reduced {len(boxes)} → {len(keep)} boxes")"""))

cells.append(md("### 4.3 Experiment — NMS Visualization"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Visualize NMS before/after                ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Generate a synthetic image with 2-3 objects        ║
# ║  2. Create many overlapping boxes per object           ║
# ║     (simulating a detector's raw output)               ║
# ║  3. Run NMS                                            ║
# ║  4. Use visualize_nms_comparison() to show before/after║
# ╚═══════════════════════════════════════════════════════╝

# Create a simple test image
test_img = torch.ones(3, 200, 300) * 0.3
# Draw "objects" as colored rectangles
test_img[:, 30:80, 40:100] = torch.tensor([0.8, 0.2, 0.2]).view(3, 1, 1)
test_img[:, 120:180, 160:250] = torch.tensor([0.2, 0.8, 0.2]).view(3, 1, 1)

# Simulate detector output (many overlapping boxes per object)
import random
all_boxes, all_scores = [], []
for cx, cy, w, h in [(70, 55, 60, 50), (205, 150, 90, 60)]:
    for _ in range(8):
        jx, jy, jw, jh = [random.gauss(0, 5) for _ in range(4)]
        all_boxes.append([cx-w//2+jx, cy-h//2+jy, cx+w//2+jw, cy+h//2+jh])
        all_scores.append(random.uniform(0.3, 0.95))

boxes_t = torch.tensor(all_boxes, dtype=torch.float)
scores_t = torch.tensor(all_scores)

keep = non_max_suppression(boxes_t, scores_t, iou_threshold=0.4)
visualize_nms_comparison(test_img, boxes_t, scores_t, keep)"""))

# ═══════════════════════════════════════════════════════
# SECTION 5 — YOLOv1 CORE LOGIC
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 5 — YOLOv1 Core Logic

## 5.1 Conceptual Background

### Grid Cell Formulation

YOLO divides the image into an S×S grid. Each cell predicts:
- **B bounding boxes**: each with (x, y, w, h, confidence)
- **C class probabilities**: shared across all B boxes

```
     ┌───┬───┬───┬───┬───┬───┬───┐
     │   │   │   │   │   │   │   │
     ├───┼───┼───┼───┼───┼───┼───┤    Each cell predicts:
     │   │   │   │   │   │   │   │     • B boxes × (x,y,w,h,conf)
     ├───┼───┼───┼───┼───┼───┼───┤     • C class probabilities
     │   │   │ 🚗│   │   │   │   │
     ├───┼───┼───┼───┼───┼───┼───┤    Object center falls in
     │   │   │   │   │   │   │   │    this cell → this cell is
     ├───┼───┼───┼───┼───┼───┼───┤    "responsible" for detecting it
     │   │   │   │   │   │   │   │
     ├───┼───┼───┼───┼───┼───┼───┤
     │   │   │   │   │   │   │   │
     └───┴───┴───┴───┴───┴───┴───┘
              S = 7
```

### Multi-Part Loss

```
L = λ_coord · L_coord + L_obj + λ_noobj · L_noobj + L_class

L_coord  = MSE of (x, y) + MSE of (√w, √h)      [only responsible boxes]
L_obj    = MSE of (conf_pred - IoU_with_GT)        [only responsible boxes]
L_noobj  = MSE of (conf_pred - 0)                  [all non-responsible boxes]
L_class  = MSE of (class_pred - class_target)       [only cells with objects]
```

### Why √(w,h)?

Small errors in large boxes matter less than in small boxes. Taking the square root compresses the range, making the loss more scale-invariant.

### Why λ_coord = 5, λ_noobj = 0.5?

- Most cells contain NO object → noobj loss dominates without downweighting
- Coordinate accuracy is critical → upweight to focus learning"""))

cells.append(md("""\
## 5.2 Implementation Tasks

Open `src/yolo_v1.py` and implement:

1. **`YOLOv1Head`** — prediction head mapping features → grid predictions
2. **`YOLOv1Loss`** — multi-part loss with coord, obj, noobj, class components
3. **`decode_predictions()`** — convert grid predictions to bounding boxes"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  SANITY CHECK: Verify YOLOv1 components                ║
# ╚═══════════════════════════════════════════════════════╝

S, B, C = 7, 2, 3

# Test YOLOv1Head
backbone_out = torch.randn(2, 256, 14, 14)  # simulated backbone output
head = YOLOv1Head(backbone_channels=256, S=S, B=B, C=C)
preds = head(backbone_out)
expected_shape = (2, S, S, B * 5 + C)
assert preds.shape == expected_shape, f"Expected {expected_shape}, got {preds.shape}"
print(f"  ✅ YOLOv1Head: {backbone_out.shape} → {preds.shape}")

# Test YOLOv1Loss
criterion = YOLOv1Loss(S=S, B=B, C=C)
targets = torch.zeros(2, S, S, 5 + C)
# Place one object in cell (3, 3)
targets[0, 3, 3, :5] = torch.tensor([0.5, 0.5, 0.2, 0.3, 1.0])
targets[0, 3, 3, 5] = 1.0  # class 0

total_loss, components = criterion(preds, targets)
assert total_loss.shape == (), f"Loss should be scalar, got {total_loss.shape}"
assert all(k in components for k in ['coord', 'obj', 'noobj', 'class']), "Missing loss components"
print(f"  ✅ YOLOv1Loss: total={total_loss.item():.4f}")
for k, v in components.items():
    print(f"      {k}: {v.item():.4f}")"""))

cells.append(md("### 5.3 Experiment — Train YOLOv1 on Synthetic Dataset"))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  EXPERIMENT: Train YOLOv1 on SimpleDetectionDataset    ║
# ║                                                       ║
# ║  TODO:                                                 ║
# ║  1. Create SimpleDetectionDataset (500 samples)        ║
# ║  2. Build a simple CNN backbone (use your PlainCNN     ║
# ║     or ResidualCNN without the final FC layer)          ║
# ║  3. Attach YOLOv1Head                                   ║
# ║  4. Train for 20-30 epochs                              ║
# ║  5. Log each loss component separately                  ║
# ║  6. Decode predictions and visualize boxes              ║
# ║  7. Apply NMS to decoded boxes                          ║
# ║                                                       ║
# ║  Hints:                                               ║
# ║  - CNN backbone: remove the avgpool + FC layers         ║
# ║  - Use a lower learning rate (1e-4 to 3e-4)            ║
# ║  - coord loss should decrease first, then class loss   ║
# ║  - If loss is NaN, check gradient clipping              ║
# ╚═══════════════════════════════════════════════════════╝

det_dataset = SimpleDetectionDataset(num_samples=500, img_size=224, S=S, C=C)
print(f"Detection dataset: {len(det_dataset)} samples")
print(f"  Image shape: {det_dataset[0][0].shape}")
print(f"  Target shape: {det_dataset[0][1].shape}")

# TODO: Build backbone + head, train, visualize
"""))

cells.append(md("""\
### 5.4 Experiment — Varying Loss Weights

Change the YOLO loss weights and observe the effect:

| Experiment | λ_coord | λ_noobj | Expected Effect |
|-----------|---------|---------|----------------|
| Default | 5.0 | 0.5 | Balanced |
| High coord | 20.0 | 0.5 | Better localization, worse classification |
| High noobj | 5.0 | 5.0 | Confidence collapses to 0 everywhere |
| No weighting | 1.0 | 1.0 | noobj dominates, poor detection |"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  TODO: Run ablation on loss weights                    ║
# ║  Compare loss curves for at least 3 configurations     ║
# ╚═══════════════════════════════════════════════════════╝
"""))

# ═══════════════════════════════════════════════════════
# SECTION 6 — DEBUGGING VISION SYSTEMS
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# Section 6 — Debugging Vision Systems

## 6.1 Common Failure Modes

| Failure | Symptom | Diagnosis | Fix |
|---------|---------|-----------|-----|
| Exploding detection loss | NaN after few steps | Grad norms spike | Gradient clipping, lower LR |
| All boxes collapse to center | Every box predicts (0.5, 0.5) | Visualize decoded boxes | Check sigmoid in loss, coord weight |
| Confidence always 0 | No detections after NMS | Plot conf distribution | Reduce λ_noobj |
| Wrong class but right box | Boxes correct, labels wrong | mAP per class | Check class head, label alignment |
| Segmentation edge bleeding | Smooth boundaries lost | Visualize close-up | Check skip connections, Dice vs CE |

## 6.2 Debugging Tasks"""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  DEBUGGING TOOLKIT                                     ║
# ║                                                       ║
# ║  TODO: For your trained YOLO model:                    ║
# ║  1. Monitor gradient norms during training              ║
# ║     (detect exploding gradients in detection loss)      ║
# ║  2. Plot activation distributions of backbone layers   ║
# ║     (check for dead neurons or saturation)              ║
# ║  3. Visualize predicted bounding boxes at different     ║
# ║     confidence thresholds (0.1, 0.3, 0.5, 0.7)         ║
# ║  4. Create an IoU histogram: for each GT box, what     ║
# ║     is the max IoU with any predicted box?              ║
# ╚═══════════════════════════════════════════════════════╝
"""))

# ═══════════════════════════════════════════════════════
# FINAL CHALLENGE
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# 🧪 Final Challenge — Vision Debug Toolkit

## Objective

Build a comprehensive **Vision Debug Toolkit** that you can reuse for any detection or segmentation project.

## Required Components

| Tool | What It Does |
|------|-------------|
| `iou_heatmap(pred_boxes, gt_boxes)` | Visualize pairwise IoU as a heatmap |
| `bbox_error_analysis(preds, gts)` | Categorize errors: localization, classification, duplicate, missing |
| `gradient_flow_backbone(model)` | Plot gradient norms through CNN backbone layers |
| `map_per_class(preds, gts, classes)` | Compute and display AP for each class |
| `segmentation_confusion(preds, gts)` | Per-class pixel confusion matrix |

## Requirements

- All functions must work on arbitrary models (not hardcoded)
- Include visualizations (matplotlib)
- Include a demo cell showing each tool on your trained models

No solutions provided — only the requirements above."""))

cells.append(code("""\
# ╔═══════════════════════════════════════════════════════╗
# ║  FINAL CHALLENGE: Implement the Vision Debug Toolkit   ║
# ║                                                       ║
# ║  Implement each function and demonstrate it on your    ║
# ║  trained models from this notebook.                    ║
# ╚═══════════════════════════════════════════════════════╝

def iou_heatmap(pred_boxes, gt_boxes, title="IoU Heatmap"):
    \"\"\"
    Compute pairwise IoU and display as a heatmap.
    TODO: Implement
    \"\"\"
    raise NotImplementedError

def bbox_error_analysis(pred_boxes, gt_boxes, iou_threshold=0.5):
    \"\"\"
    Categorize detection errors:
      - True Positive: IoU > threshold, correct class
      - Localization error: IoU 0.1-0.5, correct class
      - Classification error: IoU > threshold, wrong class
      - Duplicate: IoU > threshold but GT already matched
      - Background: IoU < 0.1 (false positive)
      - Missing: GT with no matching prediction (false negative)
    TODO: Implement
    \"\"\"
    raise NotImplementedError

def gradient_flow_backbone(model, loss):
    \"\"\"
    Plot gradient norms for each layer of a CNN backbone.
    TODO: Implement
    \"\"\"
    raise NotImplementedError

# TODO: Demonstrate each tool on your trained models
"""))

# ═══════════════════════════════════════════════════════
# SUMMARY CHECKLIST
# ═══════════════════════════════════════════════════════
cells.append(md("""\
---
# ✅ Summary Checklist

| # | Skill | Confident? |
|---|-------|-----------|
| 1 | I can build CNNs from composable ConvBlock + ResidualBlock | ☐ |
| 2 | I understand receptive fields and why 3×3 stacks work | ☐ |
| 3 | I can implement transfer learning with manual freezing | ☐ |
| 4 | I can build U-Net from scratch and train for segmentation | ☐ |
| 5 | I can implement IoU from first principles | ☐ |
| 6 | I can implement NMS without torchvision.ops | ☐ |
| 7 | I can compute mAP with 11-point interpolation | ☐ |
| 8 | I understand YOLO's grid formulation and multi-part loss | ☐ |
| 9 | I can debug vision system failures systematically | ☐ |
| 10 | I built a reusable Vision Debug Toolkit | ☐ |

---

### 🔜 Next: Group 4 — RNNs, Seq2Seq & Transformers

LSTM from scratch, attention from scratch, masking, causal decoding, KV cache, tiny GPT training."""))

# ═══════════════════════════════════════════════════════
# BUILD NOTEBOOK
# ═══════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

out_path = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, "notebooks", "03_vision_systems_lab.ipynb"
))
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
code_count = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Notebook: {out_path}")
print(f"Cells: {len(cells)} (markdown: {md_count}, code: {code_count})")
