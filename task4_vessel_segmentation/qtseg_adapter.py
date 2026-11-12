"""Adapter for the authors' FIVES-only QTSeg v0.3.1 pretrained release.

Original source and weights: https://github.com/tpnam0901/QTSeg
Definitions are imported from the preserved upstream source archive.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import urllib.request
import zipfile

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://github.com/tpnam0901/QTSeg/archive/refs/tags/v0.3.1.zip"
WEIGHTS_URL = "https://github.com/tpnam0901/QTSeg/releases/download/assets/FIVES_ckpt_v0.3.1.zip"


def ensure_assets():
    base = ROOT / "optimization" / "thirdparty" / "qtseg"
    base.mkdir(parents=True, exist_ok=True)
    source_root = base / "source" / "QTSeg-0.3.1" / "src"
    if not (source_root / "networks" / "models.py").exists():
        archive = base / "source-v0.3.1.zip"
        if not archive.exists():
            urllib.request.urlretrieve(SOURCE_URL, archive)
        destination = (base / "source").resolve()
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if not (destination / member.filename).resolve().is_relative_to(destination):
                    raise ValueError("Unsafe source archive path")
            z.extractall(destination)
    checkpoint_root = base / "checkpoint"
    weights = list(checkpoint_root.rglob("weight_best_iou.pt")) if checkpoint_root.exists() else []
    configs = list(checkpoint_root.rglob("cfg.json")) if checkpoint_root.exists() else []
    if not weights or not configs:
        archive = base / "FIVES_ckpt_v0.3.1.zip"
        if not archive.exists():
            urllib.request.urlretrieve(WEIGHTS_URL, archive)
        destination = checkpoint_root.resolve()
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if not member.filename.endswith(("cfg.json", "weight_best_iou.pt")):
                    continue
                if not (destination / member.filename).resolve().is_relative_to(destination):
                    raise ValueError("Unsafe checkpoint archive path")
                z.extract(member, destination)
        weights, configs = list(destination.rglob("weight_best_iou.pt")), list(destination.rglob("cfg.json"))
    if len(weights) != 1 or len(configs) != 1:
        raise ValueError("Expected exactly one FIVES checkpoint/config")
    return source_root, weights[0], configs[0]


def provenance():
    return {"enum": "QTSeg official FIVES v0.3.1", "url": WEIGHTS_URL,
            "source_repository": "https://github.com/tpnam0901/QTSeg/tree/v0.3.1",
            "dataset": "FIVES retinal vessels; release cfg: dataloader=FIVESA, data_root=working/dataset/FIVES",
            "normalization": "RGB [0,1] -> [-1,1]; foreground logit=class1-class0"}


class QTSegAdapter(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        source, weights, config_path = ensure_assets()
        sys.path.insert(0, str(source))
        from configs.base import Config
        from networks.models import QTSeg
        upstream = json.loads(config_path.read_text(encoding="utf-8"))
        upstream.update(encoder_pretrained="", decoder_pretrained="")
        self.network = QTSeg(Config(**upstream))
        if pretrained:
            state = torch.load(weights, map_location="cpu", weights_only=True)
            self.network.load_state_dict(state, strict=True)

    @property
    def encoder(self):
        return self.network.image_encoder

    @property
    def head(self):
        return self.network.mask_decoder.mask_tokens

    def train(self, mode=True):
        super().train(mode)
        if mode and self.cfg.get("freeze_encoder_bn"):
            for layer in self.encoder.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
        return self

    def forward(self, image):
        logits, _ = self.network(image * 2 - 1)
        return logits[:, 1:2] - logits[:, 0:1]
