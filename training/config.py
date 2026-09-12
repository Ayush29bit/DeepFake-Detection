"""Central configuration for the deepfake detection pipeline.

Defaults follow `CLAUDE.md` / `docs/architecture.md`. Everything here is
overridable from the command line (see `add_config_args`).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields

# ImageNet statistics, used because the spatial backbone is ImageNet-pretrained.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

REAL = 0
FAKE = 1
LABEL_NAMES = {REAL: "REAL", FAKE: "FAKE"}


@dataclass
class Config:
    # --- data / preprocessing ---
    image_size: int = 224
    num_frames: int = 8
    face_detector: str = "haar"          # "haar" | "mtcnn" | "none"
    face_margin: float = 0.25            # fraction of box size added around the face
    cache_dir: str = "data/processed"
    use_cache: bool = True
    train_jitter: bool = True            # jitter frame indices during training
    augment: bool = True                 # hflip / small rotation / mild colour jitter

    # --- splits ---
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    # --- model ---
    model: str = "hybrid"                # "hybrid" | "baseline"
    backbone: str = "resnet18"
    pretrained: bool = True
    freq_dim: int = 128
    d_model: int = 256
    num_heads: int = 8
    num_transformer_layers: int = 2
    dropout: float = 0.1
    use_frequency: bool = True           # ablation switch (docs/evaluation.md)
    use_temporal: bool = True            # ablation switch (docs/evaluation.md)

    # --- training ---
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 5
    num_workers: int = 0
    device: str = "auto"                 # "auto" | "cpu" | "cuda"
    threshold: float = 0.5
    best_metric: str = "f1"              # "f1" | "loss" | "accuracy" | "roc_auc"
    checkpoint_dir: str = "checkpoints"
    run_dir: str = "runs"
    run_name: str = ""

    # --- dataset location ---
    real_dir: str = ""
    fake_dir: str = ""
    manifest: str = ""
    limit: int = 0                       # cap videos per class (0 = no cap)

    extra: dict = field(default_factory=dict)

    def resolve_device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def to_dict(self) -> dict:
        return asdict(self)


_BOOL_FLAGS = {
    "pretrained", "use_cache", "train_jitter", "augment",
    "use_frequency", "use_temporal",
}


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ("1", "true", "yes", "y"):
        return True
    if value.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add one CLI flag per Config field, with the dataclass default."""
    default = Config()
    for f in fields(Config):
        if f.name == "extra":
            continue
        flag = "--" + f.name.replace("_", "-")
        current = getattr(default, f.name)
        if f.name in _BOOL_FLAGS:
            parser.add_argument(flag, type=_str2bool, default=current,
                                metavar="{true,false}")
        else:
            parser.add_argument(flag, type=type(current), default=current)
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    known = {f.name for f in fields(Config)}
    values = {k: v for k, v in vars(args).items() if k in known}
    return Config(**values)
