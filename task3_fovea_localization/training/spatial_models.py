"""Frozen public encoders, a shared spatial decoder, and pixel-correct geometry."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import timm
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from timm.models.vision_transformer import checkpoint_filter_fn

from fovea import retina_bounds, seed_everything, spatial_decode


MODELS = ("green", "dinov2", "hrnet")
INPUT_SIZE = 448


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(module):
    digest = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update((key + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def score(truth, prediction):
    distance = np.linalg.norm(np.asarray(truth) - np.asarray(prediction), axis=1)
    return dict(rmse_pixels=float(np.sqrt(np.mean(distance ** 2) / 2)),
                mean_distance=float(distance.mean()), median_distance=float(np.median(distance)),
                p90_distance=float(np.quantile(distance, .9)), max_distance=float(distance.max()),
                n=len(distance))


class Encoder(nn.Module):
    def __init__(self, name, assets, load_pretrained=True):
        super().__init__()
        if name not in MODELS:
            raise ValueError(name)
        self.name = name
        self.provenance = json.loads((Path(assets) / "source_provenance.json").read_text())[name]
        if name == "hrnet":
            self.base = timm.create_model("hrnet_w18", pretrained=False, num_classes=1000)
            self.channels = [18, 36, 72, 144]
        else:
            self.base = timm.create_model("vit_small_patch14_reg4_dinov2", pretrained=False,
                                          num_classes=0, img_size=INPUT_SIZE)
            self.channels = [384] * 4
        self.register_buffer("mean", torch.tensor(self.provenance["mean"])[None, :, None, None])
        self.register_buffer("std", torch.tensor(self.provenance["std"])[None, :, None, None])
        self.load_info = {}
        if load_pretrained:
            path = Path(assets) / self.provenance["filename"]
            if file_hash(path) != self.provenance["sha256"]:
                raise ValueError(f"Pretrained checksum mismatch: {path}")
            state = (torch.load(path, map_location="cpu", weights_only=True)
                     if path.suffix == ".pth" else load_file(str(path), device="cpu"))
            original_keys = set(state)
            native_position = list(state["pos_embed"].shape) if "pos_embed" in state else None
            if name != "hrnet":
                state = checkpoint_filter_fn(state, self.base)
            # These converted timm weights need only positional interpolation, never missing layers.
            assert set(state) == original_keys == set(self.base.state_dict())
            result = self.base.load_state_dict(state, strict=True)
            assert not result.missing_keys and not result.unexpected_keys
            self.load_info = dict(strict=True, missing_keys=[], unexpected_keys=[],
                                  native_pos_embed=native_position,
                                  position_interpolation="timm bicubic antialias to 32x32 patch grid"
                                  if name != "hrnet" else None)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images, amp=False):
        inputs = (images - self.mean) / self.std
        with torch.autocast(device_type=images.device.type, dtype=torch.float16,
                            enabled=amp and images.is_cuda):
            if self.name == "hrnet":
                e = self.base
                x = e.act1(e.bn1(e.conv1(inputs)))
                x = e.act2(e.bn2(e.conv2(x)))
                features = e.stages(x)
            else:
                features = self.base.get_intermediate_layers(
                    inputs, n=[2, 5, 8, 11], reshape=True, norm=True)
        assert len(features) == 4
        assert [f.shape[1] for f in features] == self.channels
        return [f.detach().float() for f in features]


class Block(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                         nn.GroupNorm(8, outputs), nn.SiLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                         nn.GroupNorm(8, outputs), nn.SiLU())


class SpatialHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.projections = nn.ModuleList([nn.Sequential(nn.Conv2d(c, 32, 1, bias=False),
                                         nn.GroupNorm(8, 32), nn.SiLU()) for c in channels])
        self.fuse = Block(128, 64)
        self.refine = Block(64, 32)
        self.heatmap = nn.Conv2d(32, 1, 1)
        self.region = nn.Conv2d(32, 1, 1)

    def forward(self, features):
        projected = [F.interpolate(p(f), size=(112, 112), mode="bilinear", align_corners=False)
                     for p, f in zip(self.projections, features)]
        fused = self.fuse(torch.cat(projected, 1))
        fused = F.interpolate(fused, size=(224, 224), mode="bilinear", align_corners=False)
        fused = self.refine(fused)
        return self.heatmap(fused)[:, 0], self.region(fused)[:, 0]


class SpatialModel(nn.Module):
    def __init__(self, name, assets, seed=20261021, load_pretrained=True):
        super().__init__()
        self.encoder = Encoder(name, assets, load_pretrained)
        torch.manual_seed(seed)  # Green and DINOv2 start with identical decoder weights.
        self.head = SpatialHead(self.encoder.channels)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images, amp=False):
        return self.head(self.encoder(images, amp=amp))


class ImageBank:
    def __init__(self, frame, root, split, device):
        self.frame = frame.reset_index(drop=True).copy()
        self.split = split
        self.root = Path(root)
        images, geometry = [], []
        for row in self.frame.itertuples(index=False):
            path = self.root / split / row.filename
            with Image.open(path) as source:
                assert source.size == (row.width, row.height)
                image = source.convert("RGB")
                box = retina_bounds(image)
                resized = image.crop(box).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
            images.append(np.asarray(resized).copy().transpose(2, 0, 1))
            geometry.append(dict(image_id=int(row.image_id), filename=row.filename, split=split,
                                 image_sha256=file_hash(path), left=box[0], top=box[1],
                                 right=box[2], bottom=box[3], width=row.width, height=row.height))
        self.geometry = pd.DataFrame(geometry)
        self.images = torch.from_numpy(np.stack(images)).to(device)
        self.offset = torch.tensor(self.geometry[["left", "top"]].values, device=device, dtype=torch.float32)
        self.extent = torch.tensor(self.geometry[["right", "bottom"]].values,
                                   device=device, dtype=torch.float32) - self.offset
        self.has_labels = "x" in frame
        if self.has_labels:
            assert self.frame.visible.eq(1).all(), "This experiment only validates visible landmarks"
            self.truth = torch.tensor(self.frame[["x", "y"]].values, device=device, dtype=torch.float32)
            self.xy = (self.truth - self.offset + .5) / self.extent
            boxes = torch.tensor(self.frame[["xmin", "ymin", "xmax", "ymax"]].values,
                                 device=device, dtype=torch.float32)
            self.boxes = (boxes - self.offset.repeat(1, 2) + .5) / self.extent.repeat(1, 2)

    def to_original(self, xy, indices):
        return xy * self.extent[indices] + self.offset[indices] - .5


def transform_points(points, matrix):
    ones = torch.ones_like(points[..., :1])
    homogeneous = torch.cat([points * 2 - 1, ones], -1)
    return (torch.einsum("bij,b...j->b...i", matrix, homogeneous)[..., :2] + 1) / 2


def make_batch(bank, indices, augment=False):
    device = bank.images.device
    indices = torch.as_tensor(indices, device=device, dtype=torch.long)
    images = bank.images[indices].float() / 255
    b = len(indices)
    forward = torch.eye(3, device=device).repeat(b, 1, 1)
    if augment:
        angle = torch.empty(b, device=device).uniform_(-12, 12) * np.pi / 180
        scale = torch.empty(b, device=device).uniform_(.95, 1.05)
        sign = torch.where(torch.rand(b, device=device) < .5, -1., 1.)
        forward[:, 0, 0] = angle.cos() * scale * sign
        forward[:, 0, 1] = -angle.sin() * scale
        forward[:, 1, 0] = angle.sin() * scale * sign
        forward[:, 1, 1] = angle.cos() * scale
        forward[:, :2, 2] = torch.empty(b, 2, device=device).uniform_(-.06, .06)
        theta = torch.linalg.inv(forward)
        grid = F.affine_grid(theta[:, :2], images.shape, align_corners=False)
        images = F.grid_sample(images, grid, align_corners=False, padding_mode="zeros")
        gain = torch.empty(b, 3, 1, 1, device=device).uniform_(.9, 1.1)
        brightness = torch.empty(b, 1, 1, 1, device=device).uniform_(.9, 1.1)
        gamma = torch.empty(b, 1, 1, 1, device=device).uniform_(.9, 1.1)
        images = (images * gain * brightness).clamp(0, 1).pow(gamma)
    else:
        theta = forward.clone()
    if not bank.has_labels:
        return images, theta
    xy = transform_points(bank.xy[indices], forward)
    assert ((xy >= 0) & (xy <= 1)).all(), "Augmentation moved a visible target outside the field"
    boxes = bank.boxes[indices]
    corners = torch.stack([boxes[:, [0, 1]], boxes[:, [0, 3]],
                           boxes[:, [2, 1]], boxes[:, [2, 3]]], 1)
    corners = transform_points(corners, forward)
    boxes = torch.cat([corners.min(1).values, corners.max(1).values], 1)
    return images, xy, boxes, theta


def objective(outputs, xy, boxes, theta, bank, indices):
    heat, region = outputs
    b, h, w = heat.shape
    xs = (torch.arange(w, device=heat.device) + .5) / w
    ys = (torch.arange(h, device=heat.device) + .5) / h
    dx = (xs[None, None, :] - xy[:, 0, None, None]) * w
    dy = (ys[None, :, None] - xy[:, 1, None, None]) * h
    target = torch.exp(-(dx.square() + dy.square()) / 8).flatten(1)
    target = target / target.sum(1, keepdim=True).clamp_min(1e-12)
    heat_loss = -(target * F.log_softmax(heat.float().flatten(1), 1)).sum(1).mean()
    prediction, _ = spatial_decode(heat)
    original = bank.to_original(transform_points(prediction, theta), indices)
    coordinate_loss = ((original - bank.truth[indices]) / 1000).square().mean()
    inside_x = (xs[None, None, :] >= boxes[:, 0, None, None]) & (xs[None, None, :] <= boxes[:, 2, None, None])
    inside_y = (ys[None, :, None] >= boxes[:, 1, None, None]) & (ys[None, :, None] <= boxes[:, 3, None, None])
    target_region = (inside_x & inside_y).float()
    region_loss = F.binary_cross_entropy_with_logits(region, target_region,
                                      pos_weight=torch.tensor(20., device=heat.device))
    return heat_loss + 20 * coordinate_loss + .15 * region_loss


@torch.inference_mode()
def predict(model, bank, indices, batch_size=8, head=None):
    model.eval()
    selected_head = model.head if head is None else head
    selected_head.eval()
    identity, tta = [], []
    for start in range(0, len(indices), batch_size):
        batch_ids = np.asarray(indices[start:start + batch_size])
        images = make_batch(bank, batch_ids)[0]
        # Use full precision for the saved validation and test coordinates.
        heat, _ = selected_head(model.encoder(images, amp=False))
        point, _ = spatial_decode(heat)
        heat_flip, _ = selected_head(model.encoder(images.flip(-1), amp=False))
        point_flip, _ = spatial_decode(heat_flip)
        point_flip[:, 0] = 1 - point_flip[:, 0]
        identity.append(bank.to_original(point, batch_ids).cpu().numpy())
        tta.append(bank.to_original((point + point_flip) / 2, batch_ids).cpu().numpy())
    return np.concatenate(identity), np.concatenate(tta)


def load_saved(path, assets, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = SpatialModel(saved["model_name"], assets, saved["seed"], load_pretrained=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device).eval()
    assert state_hash(model.encoder) == saved["frozen_encoder_sha256"]
    return model, saved
