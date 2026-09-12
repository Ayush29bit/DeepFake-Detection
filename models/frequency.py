"""Frequency branch: 2D FFT log-magnitude representation + lightweight CNN.

The FFT does not by itself identify manipulated video. It is an alternative
representation of the same face crop, in which the model may be able to learn
discriminative patterns that are harder to see in the pixel domain. Whether it
helps is an experimental question (docs/evaluation.md), not an assumption.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from preprocessing.dataset import denormalise

# Luminance weights (ITU-R BT.601), used for the RGB -> grayscale conversion.
_GRAY_WEIGHTS = (0.299, 0.587, 0.114)


def to_grayscale(x: torch.Tensor) -> torch.Tensor:
    """(N, 3, H, W) in [0, 1] -> (N, 1, H, W) grayscale."""
    weights = torch.tensor(_GRAY_WEIGHTS, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


def fft_log_magnitude(frames: torch.Tensor, normalised_input: bool = True,
                      eps: float = 1e-6) -> torch.Tensor:
    """Face crops -> normalised log-magnitude spectrum.

    Steps (docs/architecture.md, "Frequency Branch"):
    grayscale -> 2D FFT -> fftshift -> magnitude -> log(1 + magnitude)
    -> per-sample min-max normalisation to [0, 1].

    Input  : (N, 3, H, W). If `normalised_input`, the tensor is assumed to carry
             ImageNet normalisation and is undone first, so the spectrum is
             computed from the same pixel values a viewer would see.
    Output : (N, 1, H, W)
    """
    x = denormalise(frames) if normalised_input else frames
    gray = to_grayscale(x)
    spectrum = torch.fft.fft2(gray.to(torch.float32))
    spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
    magnitude = torch.log1p(torch.abs(spectrum))

    flat = magnitude.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (magnitude - lo) / (hi - lo + eps)


class FrequencyCNN(nn.Module):
    """Small conv stack mapping a frequency image to a fixed-size embedding.

    Deliberately lightweight (four conv+BN+ReLU+pool blocks, ~0.1M parameters):
    the spatial branch is the pretrained model, this branch is not.

    Input  : (N, in_channels, H, W)
    Output : (N, out_dim)
    """

    def __init__(self, out_dim: int = 128, in_channels: int = 1,
                 widths=(16, 32, 64, 128), dropout: float = 0.1):
        super().__init__()
        layers = []
        channels = in_channels
        for width in widths:
            layers += [
                nn.Conv2d(channels, width, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            channels = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, out_dim)
        self.out_features = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.fc(self.dropout(h))


class FrequencyBranch(nn.Module):
    """FFT representation + lightweight CNN, as one module.

    Input  : (N, 3, H, W) face crops (ImageNet-normalised by default)
    Output : (N, out_dim) frequency embedding
    """

    def __init__(self, out_dim: int = 128, normalised_input: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.cnn = FrequencyCNN(out_dim=out_dim, in_channels=1, dropout=dropout)
        self.normalised_input = normalised_input
        self.out_features = out_dim

    def spectrum(self, frames: torch.Tensor) -> torch.Tensor:
        return fft_log_magnitude(frames, normalised_input=self.normalised_input)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.cnn(self.spectrum(frames))
