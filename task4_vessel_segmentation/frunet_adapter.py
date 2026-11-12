"""Transfer the authors' CHASEDB1 FR-UNet checkpoint without DRIVE labels."""
from pathlib import Path
import hashlib
import urllib.request

import torch
from torch import nn

from fr_unet_upstream import FR_UNet

ROOT = Path(__file__).resolve().parent
URL = "https://raw.githubusercontent.com/lseventeen/FR-UNet/master/pretrained_weights/CHASEDB1/checkpoint-epoch40.pth"
EXPECTED_SIZE = 88858475
EXPECTED_SHA256 = "a451f687980e8f2b11b5922fdb05f62f6c72fb8344811a0a736409792d0eacf4"


def load_checkpoint(path):
    # Permit only an inert dictionary for the inspected legacy config class;
    # keep tensor loading restricted instead of enabling arbitrary pickle code.
    with torch.serialization.safe_globals([(dict, "bunch.Bunch")]):
        return torch.load(path, map_location="cpu", weights_only=True)


def ensure_assets():
    target = ROOT / "optimization" / "thirdparty" / "frunet" / "CHASEDB1-epoch40.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(".part")
        urllib.request.urlretrieve(URL, temporary)
        if temporary.stat().st_size != EXPECTED_SIZE:
            raise ValueError("Incomplete CHASEDB1 checkpoint download")
        temporary.replace(target)
    if target.stat().st_size != EXPECTED_SIZE:
        raise ValueError("Invalid CHASEDB1 checkpoint size")
    if hashlib.sha256(target.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Invalid CHASEDB1 checkpoint SHA256")
    return target


def provenance():
    return {"enum": "FR-UNet official CHASEDB1 epoch40", "url": URL,
            "source_repository": "https://github.com/lseventeen/FR-UNet",
            "upstream_model_blob": "bc464f5fe3e125f43c5327cd0ec51e5ecce6143d",
            "dataset": "CHASEDB1 training subjects 01..10; no DRIVE supervised checkpoint",
            "normalization": "grayscale per-image min-max to [0,1] (upstream affine Normalize cancels)",
            "license": "MIT; FR_UNET_LICENSE.txt"}


class FRUNetAdapter(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.network = FR_UNet(dropout=float(cfg.get("fr_dropout", .2)))
        if pretrained:
            checkpoint = load_checkpoint(ensure_assets())
            if "state_dict" in checkpoint:
                state = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state = checkpoint["model"]
            else:
                state = checkpoint
            state = {key.removeprefix("module."): value for key, value in state.items()}
            self.network.load_state_dict(state, strict=True)

    @property
    def encoder(self):
        return self.network

    @property
    def head(self):
        return self.network.final5

    def train(self, mode=True):
        super().train(mode)
        if mode and self.cfg.get("freeze_encoder_bn"):
            for layer in self.network.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
        return self

    def forward(self, image):
        if image.shape[1] != 1:
            raise ValueError("FR-UNet requires the declared grayscale preprocessing")
        return self.network(image)
