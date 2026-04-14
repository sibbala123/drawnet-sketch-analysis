"""
DrawNet Model
-------------
Shared ResNet-50 backbone with two task-specific heads:

  Intent head    : softmax classification over 30 intent categories
                   Loss -> CrossEntropyLoss

  Deviation head : multi-label sigmoid over 4 deviation classes
                   Loss -> BCEWithLogitsLoss with pos_weight

All images have a valid intent label — no masking required.

Backbone unfreezing schedule (controlled from train.py):
  Phase 1  epochs  1-10  : backbone fully frozen  -> only heads train
  Phase 2  epochs 11-20  : unfreeze layer4         -> fine-tune deep features
  Phase 3  epochs 21+    : unfreeze all layers     -> full fine-tuning at low LR
"""

import torch
import torch.nn as nn
from torchvision import models


class DrawNet(nn.Module):
    """
    Parameters
    ----------
    num_intent_classes : int
        Number of sketch categories (intent head output size). Default 30.
    num_deviation_classes : int
        Number of deviation types (deviation head output size). Default 4.
    pretrained : bool
        Load ImageNet weights for the backbone.
    freeze_backbone : bool
        If True, backbone weights are frozen at initialisation.
    dropout : float
        Dropout probability applied before each head's first linear layer.
    """

    def __init__(
        self,
        num_intent_classes:    int   = 30,
        num_deviation_classes: int   = 4,
        pretrained:            bool  = True,
        freeze_backbone:       bool  = True,
        dropout:               float = 0.5,
    ):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        weights        = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet         = models.resnet50(weights=weights)
        self.backbone  = nn.Sequential(*list(resnet.children())[:-1])
        # Output: (B, 2048, 1, 1) -> flatten to (B, 2048)

        if freeze_backbone:
            self._freeze_backbone()

        feat_dim = 2048

        # ── Intent head ───────────────────────────────────────────────────────
        self.intent_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_intent_classes),
        )

        # ── Deviation head ────────────────────────────────────────────────────
        self.deviation_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_deviation_classes),
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 3, H, W)

        Returns
        -------
        intent_logits    : (B, num_intent_classes)
        deviation_logits : (B, num_deviation_classes)
        """
        features         = self.backbone(x).flatten(1)
        intent_logits    = self.intent_head(features)
        deviation_logits = self.deviation_head(features)
        return intent_logits, deviation_logits

    # ── Backbone unfreezing schedule ──────────────────────────────────────────

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_phase(self, phase: int):
        """
        Progressive backbone unfreezing.
          phase 1 -> unfreeze layer4 only
          phase 2 -> unfreeze all backbone layers
        """
        if phase == 1:
            layer4 = list(self.backbone.children())[7]
            for param in layer4.parameters():
                param.requires_grad = True
            print("[DrawNet] Unfroze backbone layer4.")
        elif phase == 2:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("[DrawNet] Unfroze full backbone.")

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


# ── Multi-task loss ────────────────────────────────────────────────────────────

class DrawNetLoss(nn.Module):
    """
    Combined multi-task loss:

        L_total = lambda1 * L_intent + lambda2 * L_deviation

    Parameters
    ----------
    pos_weight : torch.Tensor, shape (num_deviation_classes,)
        Per-class positive weights for BCEWithLogitsLoss.
    lambda1 : float   weight for intent loss
    lambda2 : float   weight for deviation loss
    """

    def __init__(
        self,
        pos_weight: torch.Tensor = None,
        lambda1:    float        = 1.0,
        lambda2:    float        = 1.0,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.intent_criterion    = nn.CrossEntropyLoss()
        self.deviation_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        intent_logits:    torch.Tensor,   # (B, num_intent_classes)
        deviation_logits: torch.Tensor,   # (B, num_deviation_classes)
        intent_labels:    torch.Tensor,   # (B,) integer class indices
        deviation_labels: torch.Tensor,   # (B, num_deviation_classes) float binary
    ):
        l_intent    = self.intent_criterion(intent_logits, intent_labels)
        l_deviation = self.deviation_criterion(deviation_logits, deviation_labels)
        total       = self.lambda1 * l_intent + self.lambda2 * l_deviation
        return total, l_intent, l_deviation
