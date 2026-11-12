"""Four strictly loaded, frozen public encoders for the same spatial decoder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from safetensors.torch import load_file
from timm.layers import resample_abs_pos_embed

from spatial_models import (ImageBank, INPUT_SIZE, SpatialHead, file_hash, make_batch,
                            objective, score, seed_everything, state_hash, transform_points)
from fovea import spatial_decode


CANDIDATES = ("dinov3_convnext", "flair", "rfm", "visionfm")
MODELS = CANDIDATES


def available_models(assets):
    manifest = json.loads((Path(assets) / "source_provenance.json").read_text())
    return tuple(name for name in CANDIDATES if manifest.get(name, {}).get("status") == "READY")


def backbone(name, input_size=INPUT_SIZE):
    if name == "dinov3_convnext":
        return timm.create_model("convnext_tiny", pretrained=False, num_classes=0), [96, 192, 384, 768]
    if name == "flair":
        model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Identity()
        return model, [256, 512, 1024, 2048]
    if name == "rfm":
        # The released 64M backbone contains exactly nine blocks and no final norm.
        model = timm.create_model("vit_base_patch16_dinov3", pretrained=False, num_classes=0,
                                  depth=9, use_post_transformer_norm=False, img_size=input_size)
        return model, [768] * 4
    if name == "visionfm":
        return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0,
                                  img_size=input_size), [768] * 4
    raise ValueError(name)


class Encoder(nn.Module):
    def __init__(self, name, assets, load_pretrained=True):
        super().__init__()
        self.name = name
        self.provenance = json.loads((Path(assets) / "source_provenance.json").read_text())[name]
        assert self.provenance["status"] == "READY"
        self.base, self.channels = backbone(name)
        self.register_buffer("mean", torch.tensor(self.provenance["mean"])[None, :, None, None])
        self.register_buffer("std", torch.tensor(self.provenance["std"])[None, :, None, None])
        self.load_info = {}
        if load_pretrained:
            path = Path(assets) / self.provenance["filename"]
            assert file_hash(path) == self.provenance["sha256"]
            state = load_file(str(path), device="cpu")
            original_keys = set(state)
            position_shape = list(state["pos_embed"].shape) if "pos_embed" in state else None
            if name == "visionfm":
                # Match official VisionFM's bicubic, align_corners=False interpolation.
                state["pos_embed"] = resample_abs_pos_embed(state["pos_embed"], [28, 28],
                    num_prefix_tokens=1, interpolation="bicubic", antialias=False)
            assert original_keys == set(state) == set(self.base.state_dict())
            message = self.base.load_state_dict(state, strict=True)
            assert not message.missing_keys and not message.unexpected_keys
            self.load_info = dict(strict=True, missing_keys=[], unexpected_keys=[],
                source_original_sha256=self.provenance["original_sha256"],
                converted_encoder_sha256=self.provenance["sha256"],
                native_position_shape=position_shape,
                position_interpolation="bicubic without antialias 14x14 to 28x28" if name == "visionfm" else None,
                actual_transformer_depth=9 if name == "rfm" else 12 if name == "visionfm" else None)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images, amp=False):
        x = (images - self.mean) / self.std
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp and x.is_cuda):
            if self.name == "dinov3_convnext":
                x = self.base.stem(x)
                features = []
                for stage in self.base.stages:
                    x = stage(x)
                    features.append(x)
            elif self.name == "flair":
                e = self.base
                x = e.maxpool(e.relu(e.bn1(e.conv1(x))))
                features = []
                for stage in (e.layer1, e.layer2, e.layer3, e.layer4):
                    x = stage(x)
                    features.append(x)
            elif self.name == "rfm":
                features = self.base.forward_intermediates(x, indices=[1, 3, 5, 8],
                    intermediates_only=True, output_fmt="NCHW", norm=False, stop_early=True)
            else:
                features = self.base.get_intermediate_layers(x, n=[2, 5, 8, 11], reshape=True, norm=True)
        assert len(features) == 4 and [item.shape[1] for item in features] == self.channels
        return [item.detach().float() for item in features]


class SpatialModel(nn.Module):
    def __init__(self, name, assets, seed=20261021, load_pretrained=True):
        super().__init__()
        self.encoder = Encoder(name, assets, load_pretrained)
        torch.manual_seed(seed)
        self.head = SpatialHead(self.encoder.channels)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images, amp=False):
        return self.head(self.encoder(images, amp))


@torch.inference_mode()
def predict(model, bank, indices, batch_size=8, head=None):
    model.eval()
    head = model.head if head is None else head
    head.eval()
    identity, tta = [], []
    for start in range(0, len(indices), batch_size):
        batch_ids = np.asarray(indices[start:start + batch_size])
        images = make_batch(bank, batch_ids)[0]
        heat, _ = head(model.encoder(images, amp=False))
        point, _ = spatial_decode(heat)
        heat_flip, _ = head(model.encoder(images.flip(-1), amp=False))
        flip, _ = spatial_decode(heat_flip)
        flip[:, 0] = 1 - flip[:, 0]
        identity.append(bank.to_original(point, batch_ids).cpu().numpy().astype(np.float64))
        tta.append(bank.to_original((point + flip) / 2, batch_ids).cpu().numpy().astype(np.float64))
    return np.concatenate(identity), np.concatenate(tta)


def load_saved(path, assets, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = SpatialModel(saved["model_name"], assets, saved["seed"], load_pretrained=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device).eval()
    assert state_hash(model.encoder) == saved["frozen_encoder_sha256"]
    return model, saved
