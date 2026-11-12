"""Official Fundus VisionFM teacher with a native-grid UNETR decoder.

Encoder and decoder architecture: ABILab-CUHK/VisionFM, CC-BY-NC-4.0.
Only the public self-supervised Fundus teacher is imported, never a downstream
DRIVE segmentation checkpoint. See VISIONFM_LICENSE.txt and vendored blob IDs.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from visionfm_backbone import vit_base
from visionfm_head import Unetr_Head

ROOT = Path(__file__).resolve().parent
FILE_ID = "13uWm0a02dCWyARUcrCdHZIcEgRfBmVA4"
URL = f"https://drive.google.com/file/d/{FILE_ID}/view"
RAW_SIZE = 1507468320
RAW_SHA256 = "9a3176a0308066d6e9c9cf40fc03bfd5fa4580dd3dc62ddb2b5b8db4f2e4f350"
ENCODER_DIGEST = "b393a76b39bbed251227918ee1702dd2d22dce8fd16105be9a054df329108fda"
MEAN = [.423737496137619, .2609460651874542, .128403902053833]
STD = [.29482534527778625, .20167365670204163, .13668020069599152]


def tensor_digest(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def ensure_assets():
    base = ROOT / "optimization/thirdparty/visionfm"
    base.mkdir(parents=True, exist_ok=True)
    compact = base / "fundus_backbone.pt"
    if compact.exists():
        state = torch.load(compact, map_location="cpu", weights_only=True)
        if tensor_digest(state) != ENCODER_DIGEST:
            raise ValueError("VisionFM Fundus encoder tensor identity mismatch")
        return compact
    raw = base / "VFM_Fundus_weights.pth"
    if not raw.exists():
        try:
            import gdown
        except ModuleNotFoundError:
            sys.path.insert(0, str(ROOT / ".runtime"))
            import gdown
        result = gdown.download(id=FILE_ID, output=str(raw), quiet=True, use_cookies=False, resume=True)
        if result is None:
            raise RuntimeError("Official public Fundus checkpoint download unavailable")
    if raw.stat().st_size != RAW_SIZE or file_digest(raw) != RAW_SHA256:
        raise ValueError("VisionFM raw checkpoint identity mismatch")
    # The inspected author checkpoint contains only these additional metadata
    # types; tensor loading remains restricted, including on PyTorch >=2.6.
    allowed = [argparse.Namespace, (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
               np.dtype, type(np.dtype("float64")), type(np.dtype("float32")), type(np.dtype("int64"))]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(raw, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module.").removeprefix("backbone."): value
             for key, value in checkpoint["teacher"].items()
             if key.removeprefix("module.").startswith("backbone.")}
    if len(state) != 150 or tensor_digest(state) != ENCODER_DIGEST:
        raise ValueError("Unexpected Fundus teacher tensors")
    torch.save(state, compact)
    (base / "origin.json").write_text(json.dumps({"url": URL, "raw_sha256": RAW_SHA256,
        "encoder_tensor_digest": ENCODER_DIGEST, "selection": "teacher.backbone only; excludes student/optimizer/head"}, indent=2), encoding="utf-8")
    return compact


def provenance(cfg=None):
    result = {"enum": "VisionFM official Fundus self-supervised teacher", "url": URL,
            "raw_sha256": RAW_SHA256, "encoder_tensor_digest": ENCODER_DIGEST,
            "source_repository": "https://github.com/ABILab-CUHK/VisionFM",
            "supervision": "foundation encoder only; new UNETR head trained on each fold's own labels",
            "normalization": {"mean": MEAN, "std": STD}, "license": "CC-BY-NC-4.0; VISIONFM_LICENSE.txt"}
    if cfg and cfg.get("encoder") == "visionfm_hybrid":
        result.update(decoder="new multiscale CNN/transformer feature fuser",
                      supervision="foundation encoder only; new multiscale vessel head trained on each fold's own labels",
                      local_feature_pretraining={"enum": "ResNet18 IMAGENET1K_V1 stem/layer1/layer2",
                                                 "url": models.ResNet18_Weights.IMAGENET1K_V1.url})
    if cfg and cfg.get("task") == "external_pretrain":
        result["supervision"] = "public Fundus teacher; dense vessel head learned only from the external FIVES training split"
    return result


class VisionFMUNet(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.encoder = vit_base(img_size=[224], patch_size=16, num_classes=0, use_flash=True)
        if pretrained:
            state = torch.load(ensure_assets(), map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(state, strict=True)
        self.decoder = Unetr_Head(768, 1, img_dim=cfg["size"], dropout_rate=float(cfg.get("dropout", .3)))
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        if cfg.get("freeze_encoder", True):
            self.encoder.requires_grad_(False)

    @property
    def head(self):
        return self.decoder.cls

    def train(self, mode=True):
        super().train(mode)
        if mode and self.cfg.get("freeze_encoder", True):
            self.encoder.eval()
        return self

    def forward(self, image):
        if image.shape[-1] % 16 or image.shape[-2] % 16:
            raise ValueError("VisionFM input must be divisible by 16")
        normalized = (image - self.mean) / self.std
        tokens = self.encoder.get_intermediate_layers(normalized, 12)
        return self.decoder([tokens[index][:, 1:] for index in [3, 5, 7, 11]], normalized)


class DetailBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))


class VisionFMHybrid(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.encoder = vit_base(img_size=[224], patch_size=16, num_classes=0, use_flash=True)
        if pretrained:
            self.encoder.load_state_dict(torch.load(ensure_assets(), map_location="cpu", weights_only=True), strict=True)
        if cfg.get("freeze_encoder", True):
            self.encoder.requires_grad_(False)
        cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.cnn_encoder = nn.ModuleList([nn.Sequential(cnn.conv1, cnn.bn1, cnn.relu),
                                         nn.Sequential(cnn.maxpool, cnn.layer1), cnn.layer2])
        self.detail = DetailBlock(3, 16)
        self.token_projections = nn.ModuleList([nn.Conv2d(768, 64, 1) for _ in range(4)])
        self.token_fuse = DetailBlock(256, 128)
        self.decoders = nn.ModuleList([DetailBlock(128 + 128, 96), DetailBlock(96 + 64, 64),
                                      DetailBlock(64 + 64, 32), DetailBlock(32 + 16, 24)])
        self.dropout = nn.Dropout2d(float(cfg.get("dropout", .05)))
        self.head = nn.Conv2d(24, 1, 1)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.register_buffer("cnn_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("cnn_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def pretrained_parameters(self):
        return list(self.encoder.parameters()) + list(self.cnn_encoder.parameters())

    def train(self, mode=True):
        super().train(mode)
        if mode and self.cfg.get("freeze_encoder", True):
            self.encoder.eval()
        return self

    def forward(self, image):
        height, width = image.shape[-2:]
        if height % 16 or width % 16:
            raise ValueError("VisionFM dimensions must be divisible by 16")
        tokens = self.encoder.get_intermediate_layers((image - self.mean) / self.std, 12)
        maps = [tokens[index][:, 1:].transpose(1, 2).reshape(image.shape[0], 768, height // 16, width // 16)
                for index in [3, 5, 7, 11]]
        x = self.token_fuse(torch.cat([layer(value) for layer, value in zip(self.token_projections, maps)], 1))
        full = self.detail(image * 2 - 1)
        half = self.cnn_encoder[0]((image - self.cnn_mean) / self.cnn_std)
        quarter = self.cnn_encoder[1](half)
        eighth = self.cnn_encoder[2](quarter)
        for decoder, skip in zip(self.decoders, [eighth, quarter, half, full]):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], 1))
        return self.head(self.dropout(x))
