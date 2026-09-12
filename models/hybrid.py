"""Hybrid model: spatial + frequency fusion per frame, then temporal Transformer.

Pipeline for one batch of videos (B, T, 3, H, W):

    spatial   ResNet18            -> (B, T, 512)
    frequency FFT + small CNN     -> (B, T, 128)
    fusion    concat + Linear     -> (B, T, 256)
    temporal  TransformerEncoder  -> (B, T, 256)
    pooling   mean over frames    -> (B, 256)
    head      Linear(256, 1)      -> (B,) logit

The Transformer here is `torch.nn.TransformerEncoder` used as a sequence model
over 8 numeric frame embeddings. It is not a language model: no tokenizer, no
vocabulary, no text.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.baseline import SpatialBranch
from models.frequency import FrequencyBranch


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding, so frame order is represented.

    Input/Output: (B, T, D)
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(x.dtype)


class HybridDeepfakeDetector(nn.Module):
    """Spatial-frequency-temporal detector producing one logit per video.

    `use_frequency` and `use_temporal` exist to run the three ablation
    configurations defined in docs/evaluation.md:
      1. spatial only                  (use_frequency=False, use_temporal=False)
      2. spatial + frequency           (use_frequency=True,  use_temporal=False)
      3. spatial + frequency + temporal (both True - the full proposed model)
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 freq_dim: int = 128, d_model: int = 256, num_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.1,
                 use_frequency: bool = True, use_temporal: bool = True,
                 num_frames: int = 8):
        super().__init__()
        self.use_frequency = use_frequency
        self.use_temporal = use_temporal
        self.d_model = d_model

        self.spatial = SpatialBranch(backbone=backbone, pretrained=pretrained)
        fused_dim = self.spatial.out_features
        if use_frequency:
            self.frequency = FrequencyBranch(out_dim=freq_dim, dropout=dropout)
            fused_dim += freq_dim
        else:
            self.frequency = None

        self.fusion = nn.Linear(fused_dim, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if use_temporal:
            self.pos_encoder = PositionalEncoding(d_model, max_len=max(num_frames, 512))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model,
                dropout=dropout, batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                     enable_nested_tensor=False)
        else:
            self.pos_encoder = None
            self.transformer = None

        self.head = nn.Linear(d_model, 1)

    # ------------------------------------------------------------- forward
    def frame_embeddings(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) -> (B, T, d_model) fused per-frame embeddings."""
        if frames.dim() != 5:
            raise ValueError(f"expected (B, T, 3, H, W), got {tuple(frames.shape)}")
        b, t = frames.shape[:2]
        flat = frames.flatten(0, 1)

        spatial_embed = self.spatial(flat)                      # (B*T, 512)
        if self.frequency is not None:
            freq_embed = self.frequency(flat)                   # (B*T, 128)
            fused = torch.cat([spatial_embed, freq_embed], dim=1)
        else:
            fused = spatial_embed

        projected = self.fusion_norm(self.fusion(self.dropout(fused)))
        return projected.view(b, t, self.d_model)

    def sequence_features(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) -> (B, T, d_model) after temporal modelling."""
        seq = self.frame_embeddings(frames)
        if self.transformer is not None:
            seq = self.transformer(self.pos_encoder(seq))
        return seq

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        pooled = self.sequence_features(frames).mean(dim=1)     # (B, d_model)
        return self.head(self.dropout(pooled)).squeeze(-1)      # (B,)


def build_model(config) -> nn.Module:
    """Construct the model named by `config.model` ("hybrid" or "baseline")."""
    from models.baseline import ResNetBaseline

    if config.model == "baseline":
        return ResNetBaseline(backbone=config.backbone, pretrained=config.pretrained,
                              dropout=config.dropout)
    if config.model == "hybrid":
        return HybridDeepfakeDetector(
            backbone=config.backbone, pretrained=config.pretrained,
            freq_dim=config.freq_dim, d_model=config.d_model,
            num_heads=config.num_heads, num_layers=config.num_transformer_layers,
            dropout=config.dropout, use_frequency=config.use_frequency,
            use_temporal=config.use_temporal, num_frames=config.num_frames)
    raise ValueError(f"unknown model {config.model!r}; expected 'hybrid' or 'baseline'")
