"""Shared image geometry, model, and inference for fovea localization.

All training labels come from the supplied competition training split.
ImageNet initialization is optional; no external retinal labels are used.
"""
from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, resnet34


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def retina_bounds(image):
    """Find the largest nonblack component; retain 1% margin around the retina."""
    w, h = image.size
    small = np.asarray(image.resize((256, 256)))
    mask = (small[:, :, 0] > 12) & (small[:, :, 1] > 5)
    labels, count = ndimage.label(mask)
    if count == 0:
        return (0, 0, w, h)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    y, x = ndimage.find_objects(labels, max_label=count)[int(sizes.argmax()) - 1]
    return (max(0, int(x.start / 256 * w - w * .01)),
            max(0, int(y.start / 256 * h - h * .01)),
            min(w, int(x.stop / 256 * w + w * .01)),
            min(h, int(y.stop / 256 * h + h * .01)))


class ImageBank:
    """Cache RGB fields of view. Labels are never needed for image preprocessing."""
    def __init__(self, frame, data_root, split, size, device):
        self.frame = frame.reset_index(drop=True).copy()
        images, bounds = [], []
        for row in self.frame.itertuples():
            with Image.open(Path(data_root) / split / row.filename) as source:
                image = source.convert("RGB")
                box = retina_bounds(image)
                image = image.crop(box).resize((size, size), Image.Resampling.BILINEAR)
                images.append(np.asarray(image).copy().transpose(2, 0, 1))
                bounds.append(box)
        self.images = torch.from_numpy(np.stack(images)).to(device)
        self.bounds = np.array(bounds, dtype=np.float64)
        self.offset = self.bounds[:, :2]
        self.extent = self.bounds[:, 2:] - self.offset
        if "x" in self.frame:
            xy = self.frame[["x", "y"]].values
            self.xy = torch.tensor((xy - self.offset) / self.extent, dtype=torch.float32, device=device)
            self.visible = torch.tensor(self.frame.visible.values.astype(bool), device=device)
            boxes = self.frame[["xmin", "ymin", "xmax", "ymax"]].values
            normalized = (boxes - np.tile(self.offset, (1, 2))) / np.tile(self.extent, (1, 2))
            self.boxes = torch.tensor(normalized, dtype=torch.float32, device=device)

    def to_original(self, xy, indices=None):
        ids = np.arange(len(self.frame)) if indices is None else np.asarray(indices)
        return np.asarray(xy) * self.extent[ids] + self.offset[ids]

    def from_original(self, xy, indices=None):
        ids = np.arange(len(self.frame)) if indices is None else np.asarray(indices)
        return (np.asarray(xy) - self.offset[ids]) / self.extent[ids]


class ConvBlock(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(8, outputs), nn.SiLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(8, outputs), nn.SiLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, inputs, skip, outputs):
        super().__init__()
        self.block = ConvBlock(inputs + skip, outputs)

    def forward(self, x, skip):
        return self.block(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))


class FoveaNet(nn.Module):
    """ResNet18 encoder, U-Net decoder, landmark and auxiliary region heads.

    An additional logit represents 'not visible'. It is part of the spatial
    classification loss and supports real (0,0) labels if such examples exist.
    The supplied training set has no such examples: absence is NOT validated.
    """
    def __init__(self, encoder_weights=None, backbone="resnet18"):
        super().__init__()
        if backbone not in ("resnet18", "resnet34"):
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.encoder = {"resnet18": resnet18, "resnet34": resnet34}[backbone](weights=None)
        if encoder_weights:
            path = Path(encoder_weights)
            if path.name == "resnet18-f37072fd.pth":
                assert hashlib.sha256(path.read_bytes()).hexdigest().startswith("f37072fd"), "Weight checksum mismatch"
            if path.name == "resnet34-b627a593.pth":
                assert hashlib.sha256(path.read_bytes()).hexdigest().startswith("b627a593"), "Weight checksum mismatch"
            self.encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.encoder.fc = nn.Identity()
        self.d3 = DecoderBlock(512, 256, 128)
        self.d2 = DecoderBlock(128, 128, 96)
        self.d1 = DecoderBlock(96, 64, 64)
        self.d0 = DecoderBlock(64, 64, 32)
        self.heatmap = nn.Conv2d(32, 1, 1)
        self.region = nn.Conv2d(32, 1, 1)
        self.absent = nn.Linear(512, 1)
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def train(self, mode=True):
        super().train(mode)
        # Tiny batches: use pretrained BN running statistics in the encoder.
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x):
        x = (x - self.rgb_mean) / self.rgb_std
        e = self.encoder
        s0 = e.relu(e.bn1(e.conv1(x)))
        s1 = e.layer1(e.maxpool(s0))
        s2 = e.layer2(s1)
        s3 = e.layer3(s2)
        s4 = e.layer4(s3)
        z = self.d0(self.d1(self.d2(self.d3(s4, s3), s2), s1), s0)
        return self.heatmap(z)[:, 0], self.region(z)[:, 0], self.absent(s4.mean((2, 3)))[:, 0]


def spatial_decode(heat, absent=None):
    """Expected coordinates, in [0,1], using pixel centers and float32 softmax."""
    b, h, w = heat.shape
    prob = heat.float().flatten(1).softmax(1).view(b, h, w)
    xs = (torch.arange(w, device=heat.device) + .5) / w
    ys = (torch.arange(h, device=heat.device) + .5) / h
    xy = torch.stack([(prob.sum(1) * xs).sum(1), (prob.sum(2) * ys).sum(1)], 1)
    p_absent = torch.zeros(b, device=heat.device) if absent is None else torch.sigmoid(absent.float() - heat.float().flatten(1).logsumexp(1))
    return xy, p_absent


def objective(outputs, xy, boxes, visible, sigma=2.0):
    heat, region, absent = outputs
    b, h, w = heat.shape
    xs = (torch.arange(w, device=heat.device) + .5) / w
    ys = (torch.arange(h, device=heat.device) + .5) / h
    dx = (xs[None, None, :] - xy[:, 0, None, None]) * w
    dy = (ys[None, :, None] - xy[:, 1, None, None]) * h
    gaussian = torch.exp(-(dx.square() + dy.square()) / (2 * sigma**2)).flatten(1)
    gaussian = gaussian / gaussian.sum(1, keepdim=True).clamp_min(1e-12)
    target = torch.cat([gaussian * visible[:, None], (~visible)[:, None]], 1)
    logits = torch.cat([heat.float().flatten(1), absent.float()[:, None]], 1)
    location_loss = -(target * F.log_softmax(logits, dim=1)).sum(1).mean()
    predicted, _ = spatial_decode(heat)
    if visible.any():
        coordinate_loss = (predicted[visible] - xy[visible]).square().mean()
    else:
        coordinate_loss = predicted.sum() * 0
    inside_x = (xs[None, None, :] >= boxes[:, 0, None, None]) & (xs[None, None, :] <= boxes[:, 2, None, None])
    inside_y = (ys[None, :, None] >= boxes[:, 1, None, None]) & (ys[None, :, None] <= boxes[:, 3, None, None])
    mask = (inside_x & inside_y & visible[:, None, None]).float()
    region_loss = F.binary_cross_entropy_with_logits(region.float(), mask, pos_weight=torch.tensor(20., device=heat.device))
    return location_loss + 100 * coordinate_loss + .15 * region_loss


def make_batch(bank, indices, size, augment=False, centers=None, crop_fraction=.4):
    """Apply one affine transform consistently to image, point, and box corners.

    Fine training centers use TRAIN labels with random jitter. Validation and
    test centers must be predictions, never their ground-truth coordinates.
    """
    device = bank.images.device
    ids = torch.as_tensor(indices, dtype=torch.long, device=device)
    images = bank.images[ids].float().div_(255)
    b = len(ids)
    if centers is None:
        theta = torch.eye(3, device=device).repeat(b, 1, 1)
    else:
        # Output-to-input map for an axis-aligned crop of the field of view.
        theta = torch.eye(3, device=device).repeat(b, 1, 1)
        center = torch.as_tensor(centers, dtype=torch.float32, device=device)
        theta[:, 0, 0] = theta[:, 1, 1] = crop_fraction
        theta[:, :2, 2] = center * 2 - 1
    if augment:
        angle = torch.empty(b, device=device).uniform_(-15, 15) * np.pi / 180
        scale = torch.empty(b, device=device).uniform_(.90, 1.10)
        sign = torch.where(torch.rand(b, device=device) < .5, -1., 1.)
        forward = torch.eye(3, device=device).repeat(b, 1, 1)
        forward[:, 0, 0] = angle.cos() * scale * sign
        forward[:, 0, 1] = -angle.sin() * scale
        forward[:, 1, 0] = angle.sin() * scale * sign
        forward[:, 1, 1] = angle.cos() * scale
        forward[:, :2, 2] = torch.empty(b, 2, device=device).uniform_(-.12, .12)
        theta = theta @ torch.linalg.inv(forward)
    grid = F.affine_grid(theta[:, :2], (b, 3, size, size), align_corners=False)
    images = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if augment:
        gains = torch.empty(b, 3, 1, 1, device=device).uniform_(.85, 1.15)
        bright = torch.empty(b, 1, 1, 1, device=device).uniform_(.8, 1.2)
        gamma = torch.empty(b, 1, 1, 1, device=device).uniform_(.85, 1.18)
        images = ((images * gains * bright).clamp(0, 1) + 1e-6).pow(gamma)
        images = (images + torch.randn_like(images) * .004).clamp_(0, 1)
    if not hasattr(bank, "xy"):
        return images, theta
    inverse = torch.linalg.inv(theta)
    original_xy = torch.cat([bank.xy[ids] * 2 - 1, torch.ones(b, 1, device=device)], 1)
    xy = (torch.bmm(inverse, original_xy[:, :, None])[:, :2, 0] + 1) / 2
    boxes = bank.boxes[ids]
    corners = torch.stack([boxes[:, [0, 1]], boxes[:, [0, 3]], boxes[:, [2, 1]], boxes[:, [2, 3]]], 1)
    corners = torch.cat([corners * 2 - 1, torch.ones(b, 4, 1, device=device)], 2)
    mapped = torch.bmm(corners, inverse.transpose(1, 2))[:, :, :2] / 2 + .5
    mapped_boxes = torch.cat([mapped.min(1).values, mapped.max(1).values], 1)
    visible = bank.visible[ids] & (xy >= 0).all(1) & (xy <= 1).all(1)
    return images, xy, mapped_boxes, visible, theta


@torch.inference_mode()
def predict_bank(model, bank, indices, size, batch_size=8, tta=True, centers=None, crop_fraction=.4):
    model.eval()
    coordinates, absence = [], []
    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start:start + batch_size]
        c = None if centers is None else centers[start:start + batch_size]
        result = make_batch(bank, batch_ids, size, centers=c, crop_fraction=crop_fraction)
        images, theta = result[0], result[-1]
        values, pa = [], []
        for flip in ([False, True] if tta else [False]):
            inputs = images.flip(-1) if flip else images
            with torch.autocast(device_type=images.device.type, enabled=images.is_cuda):
                heat, _, absent = model(inputs)
            xy, p_absent = spatial_decode(heat, absent)
            if flip:
                xy[:, 0] = 1 - xy[:, 0]
            values.append(xy)
            pa.append(p_absent)
        xy = torch.stack(values).mean(0)
        homogeneous = torch.cat([xy * 2 - 1, torch.ones(len(xy), 1, device=xy.device)], 1)
        original = (torch.bmm(theta, homogeneous[:, :, None])[:, :2, 0] + 1) / 2
        coordinates.append(original.cpu().numpy())
        absence.append(torch.stack(pa).mean(0).cpu().numpy())
    return np.concatenate(coordinates), np.concatenate(absence)


class EMA:
    def __init__(self, model, decay=.97):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for a, b in zip(self.model.parameters(), model.parameters()):
            a.lerp_(b.detach(), 1 - self.decay)
        for a, b in zip(self.model.buffers(), model.buffers()):
            a.copy_(b)


def rmse(truth, predicted):
    """Competition RMSE across all 2*N coordinate entries, in original pixels."""
    return float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(predicted))**2)))


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = FoveaNet(backbone=checkpoint["config"].get("backbone", "resnet18"))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint["config"]
