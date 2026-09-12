"""Spatial branch (torchvision backbone) and the ResNet18-only baseline model.

The baseline is the comparison point for the hybrid model: same input, same
loss, same metrics, but no frequency branch and no Transformer
(docs/architecture.md, "Baseline (ResNet18-only)").
"""

from __future__ import annotations

import os
import warnings

import torch
import torch.nn as nn
import torchvision

# Feature width of each supported backbone's penultimate layer.
BACKBONE_FEATURES = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}


def _ensure_ssl_certificates() -> None:
    """Point urllib at certifi's CA bundle so pretrained weights can download.

    Without this, a machine whose system trust store is incomplete fails the
    torchvision weight download with CERTIFICATE_VERIFY_FAILED.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


class SpatialBranch(nn.Module):
    """Pretrained CNN backbone with its ImageNet classifier removed.

    Input  : (N, 3, H, W)
    Output : (N, out_features) pooled spatial embedding (512 for ResNet18)
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if backbone not in BACKBONE_FEATURES:
            raise ValueError(f"unsupported backbone {backbone!r}; "
                             f"choose from {sorted(BACKBONE_FEATURES)}")
        self.backbone_name = backbone
        self.out_features = BACKBONE_FEATURES[backbone]

        weights = None
        self.pretrained = False
        if pretrained:
            _ensure_ssl_certificates()
            try:
                weights = torchvision.models.get_weight(
                    {"resnet18": "ResNet18_Weights.IMAGENET1K_V1",
                     "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
                     "resnet50": "ResNet50_Weights.IMAGENET1K_V2"}[backbone])
                net = torchvision.models.get_model(backbone, weights=weights)
                self.pretrained = True
            except Exception as exc:
                warnings.warn(
                    f"could not load pretrained {backbone} weights ({exc}); "
                    f"falling back to random initialisation. Results from a "
                    f"randomly initialised backbone are not comparable to "
                    f"pretrained ones.", RuntimeWarning)
                net = torchvision.models.get_model(backbone, weights=None)
        else:
            net = torchvision.models.get_model(backbone, weights=None)

        net.fc = nn.Identity()   # drop the 1000-way ImageNet classifier
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNetBaseline(nn.Module):
    """Frame-level spatial baseline.

    Each sampled frame goes through the backbone and a shared linear head; the
    video-level logit is the mean of the per-frame logits.

    Input  : (B, T, 3, H, W)
    Output : (B,) video-level logit
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.spatial = SpatialBranch(backbone=backbone, pretrained=pretrained)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.spatial.out_features, 1)

    def frame_logits(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) -> (B, T) per-frame logits."""
        if frames.dim() != 5:
            raise ValueError(f"expected (B, T, 3, H, W), got {tuple(frames.shape)}")
        b, t = frames.shape[:2]
        flat = frames.flatten(0, 1)
        feats = self.spatial(flat)
        logits = self.head(self.dropout(feats))
        return logits.view(b, t)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.frame_logits(frames).mean(dim=1)
