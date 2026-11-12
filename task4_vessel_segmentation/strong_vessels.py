"""Pretrained retinal vessel experiments with fixed image-level 5-fold OOF."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import models

try:
    import cv2
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / ".runtime"))
    import cv2

from retinal import image_paths, index_images, write_json
from vessel_model import get_split, read_label, dice_per_image, choose_threshold, setup, amp_context
from competition_protocol import SCRIPT_PROTOCOL, LEGACY_PROTOCOL, FIVEFOLD, LOO, read_target, read_targets, split_for


DEFAULT = {"name": "resnet34_green_v1", "encoder": "resnet34", "preprocess": "green_clahe",
           "size": 608, "patch": 256, "batch": 8, "epochs": 100, "steps": 40,
           "lr": 0.0015, "encoder_lr_factor": 0.15, "weight_decay": 0.0001,
           "dice_weight": 0.6, "ema_decay": 0.99, "validate_every": 4,
           "seed": 42, "clahe_clip": 2.0, "tta": 4, "dropout": 0.05,
           "freeze_encoder_bn": False, "loss": "bce_dice"}


def config_from(path):
    config = dict(DEFAULT)
    if path:
        config.update(json.loads(Path(path).read_text(encoding="utf-8")))
    stride = 8 if config["encoder"] in ("frunet_chase", "retfound_green") else 32
    if config["size"] % stride or config["patch"] % stride:
        raise ValueError(f"size and patch must be multiples of {stride}")
    if config["patch"] > config["size"] or min(config["epochs"], config["steps"], config["batch"]) < 1:
        raise ValueError("Invalid training dimensions/counts")
    if config.get("native_grid") and (config.get("label_space") != "native" or
            (config["preprocess"].startswith("vascx") and config["preprocess"] != "vascx_native")):
        raise ValueError("Native-grid training requires native labels and an unwarped image preprocessor")
    if config["preprocess"] == "vascx_native" and not config.get("native_grid"):
        raise ValueError("Native VascX preprocessing requires native-grid supervision")
    if config["encoder"] == "retfound_green" and (config["size"] % 14 or config["patch"] % 14):
        raise ValueError("RETFound-Green size/patch must also be divisible by 14")
    if config.get("cv_protocol") == LOO:
        if config.get("checkpoint_selection") != "fixed_final_ema" or config.get("evaluation_protocol") != SCRIPT_PROTOCOL:
            raise ValueError("LOO must use fixed final EMA and the supplied-converter target protocol")
    if config.get("label_space") == "script512" and config.get("evaluation_protocol") != SCRIPT_PROTOCOL:
        raise ValueError("Script supervision requires matching script evaluation")
    return config


def native_padding(height, width, cfg):
    delta_h, delta_w = cfg["size"] - height, cfg["size"] - width
    if min(delta_h, delta_w) < 0:
        raise ValueError("Native image exceeds declared padding size")
    top = delta_h // 2 if cfg.get("center_pad") else 0
    left = delta_w // 2 if cfg.get("center_pad") else 0
    return (top, delta_h - top), (left, delta_w - left)


def cv_preprocess(path, cfg):
    if cfg["preprocess"] == "vascx_native":
        from vascx_adapter import prepare_native
        return prepare_native(str(path), cfg["size"], bool(cfg.get("center_pad")))
    if cfg["preprocess"] == "vascx_crop":
        from vascx_adapter import prepare_image
        return prepare_image(str(path), cfg["size"])[0]
    if cfg["preprocess"] == "vascx_resize":
        from vascx_adapter import prepare_resize
        return prepare_resize(str(path), cfg["size"])
    original = Image.open(path).convert("RGB")
    if cfg.get("native_grid"):
        rgb = np.asarray(original)
        height, width = rgb.shape[:2]
        if max(height, width) > cfg["size"]:
            raise ValueError("Native image does not fit declared padding size")
        pad_h, pad_w = native_padding(height, width, cfg)
        rgb = np.pad(rgb, (pad_h, pad_w, (0, 0)))
    else:
        rgb = np.asarray(original.resize((cfg["size"], cfg["size"]), Image.Resampling.BILINEAR))
    clahe = cv2.createCLAHE(clipLimit=float(cfg["clahe_clip"]), tileGridSize=(8, 8))
    mode = cfg["preprocess"]
    if mode == "fr_gray":
        gray_image = Image.open(path).convert("L")
        if cfg.get("native_grid"):
            gray = np.asarray(gray_image, dtype=np.float32)
            pad_h, pad_w = native_padding(*gray.shape, cfg)
            gray = np.pad(gray, (pad_h, pad_w))
        else:
            gray = np.asarray(gray_image.resize((cfg["size"], cfg["size"]), Image.Resampling.BILINEAR), dtype=np.float32)
        gray = (gray - gray.min()) / max(float(gray.max() - gray.min()), 1.)
        return np.ascontiguousarray(gray[None])
    elif mode == "green_clahe":
        green = clahe.apply(np.ascontiguousarray(rgb[..., 1]))
        rgb = np.repeat(green[..., None], 3, axis=-1)
    elif mode == "lab_clahe":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    elif mode == "retinex_clahe":
        green = rgb[..., 1].astype(np.float32) / 255
        background = ndimage.gaussian_filter(green, 12)
        corrected = np.clip((green - background) * 4 + 0.5, 0, 1)
        local = clahe.apply((corrected * 255).astype(np.uint8))
        rgb = np.stack([clahe.apply(rgb[..., 1]), local, rgb[..., 1]], -1)
    elif mode != "rgb":
        raise ValueError(f"Unknown preprocessing: {mode}")
    return np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255


def fov512(path):
    rgb = np.asarray(Image.open(path).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    return ndimage.binary_dilation(ndimage.binary_fill_holes(ndimage.gaussian_filter(rgb[..., 1], 1) > 0.02), iterations=2)


def load_data(root, ids, cfg):
    paths = index_images(Path(root) / "train/image")
    labels = index_images(Path(root) / "train/label")
    x = np.stack([cv_preprocess(paths[key], cfg) for key in ids])
    if cfg.get("native_grid"):
        native_labels = []
        for key in ids:
            label = (np.asarray(Image.open(labels[key]).convert("L")) >= 128).astype(np.float32)
            if label.shape != (Image.open(paths[key]).height, Image.open(paths[key]).width):
                raise ValueError("Native image/label shape mismatch")
            pad_h, pad_w = native_padding(*label.shape, cfg)
            native_labels.append(np.pad(label, (pad_h, pad_w))[None])
        return x, np.stack(native_labels)
    if cfg["preprocess"] == "vascx_crop":
        from vascx_adapter import prepare_image, warp_label
        y = np.stack([warp_label(read_label(labels[key], 512),
            prepare_image(str(paths[key]), cfg["size"])[1])[None] for key in ids])
        return x, y
    if cfg.get("label_space") == "script512":
        y = np.stack([np.asarray(Image.fromarray(read_target(labels[key]).astype(np.uint8)).resize(
            (cfg["size"], cfg["size"]), Image.Resampling.NEAREST), dtype=np.float32)[None] for key in ids])
    elif cfg.get("label_space") == "canonical512" and cfg["size"] != 512:
        y = np.stack([np.asarray(Image.fromarray(read_label(labels[key], 512)).resize(
            (cfg["size"], cfg["size"]), Image.Resampling.NEAREST))[None] for key in ids])
    else:
        y = np.stack([read_label(labels[key], cfg["size"])[None] for key in ids])
    return x, y


def prediction_geometries(paths, ids, cfg):
    if cfg.get("native_grid"):
        output = []
        for key in ids:
            image = Image.open(paths[key])
            pad_h, pad_w = native_padding(image.height, image.width, cfg)
            output.append({"kind": "native_pixels", "height": image.height, "width": image.width,
                           "top": pad_h[0], "left": pad_w[0]})
        return output
    if cfg["preprocess"] == "vascx_crop":
        from vascx_adapter import prepare_image
        geometries = [dict(prepare_image(str(paths[key]), cfg["size"])[1]) for key in ids]
        if cfg.get("inference_projection") == "native_first":
            for geometry in geometries:
                geometry["kind"] = "crop_native_first"
        return geometries
    if cfg.get("inference_projection") == "native_first":
        return [{"kind": "resized_native_first", "height": Image.open(paths[key]).height,
                 "width": Image.open(paths[key]).width} for key in ids]
    return [None for _ in ids]


class Block(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs),
                         nn.SiLU(inplace=True), nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                         nn.BatchNorm2d(outputs), nn.SiLU(inplace=True))


def encoder_source(name):
    if name == "resnet34":
        return models.ResNet34_Weights.IMAGENET1K_V1
    if name == "resnet18":
        return models.ResNet18_Weights.IMAGENET1K_V1
    if name == "convnext_tiny":
        return models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    if name == "efficientnet_b0":
        return models.EfficientNet_B0_Weights.IMAGENET1K_V1
    raise ValueError(name)


class PretrainedUNet(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg, self.encoder_name = cfg, cfg["encoder"]
        weights = encoder_source(self.encoder_name) if pretrained else None
        if self.encoder_name in ("resnet34", "resnet18"):
            self.encoder = getattr(models, self.encoder_name)(weights=weights)
            self.encoder.fc = nn.Identity()
            self.encoder.avgpool = nn.Identity()
            widths = [16, 64, 64, 128, 256, 512]
        elif self.encoder_name == "efficientnet_b0":
            self.encoder = models.efficientnet_b0(weights=weights).features[:-1]
            widths = [16, 16, 24, 40, 112, 320]
        else:
            self.encoder = models.convnext_tiny(weights=weights).features
            self.half_detail = Block(3, 32)
            widths = [16, 32, 96, 192, 384, 768]
        self.detail = Block(3, 16)
        decoder_widths = [192, 128, 64, 32, 24]
        self.decoders = nn.ModuleList()
        channels = widths[-1]
        for skip_channels, outputs in zip(reversed(widths[:-1]), decoder_widths):
            self.decoders.append(Block(channels + skip_channels, outputs))
            channels = outputs
        self.dropout = nn.Dropout2d(float(cfg["dropout"]))
        self.head = nn.Conv2d(channels, 1, 1)
        self.register_buffer("input_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if mode and self.cfg.get("freeze_encoder_bn"):
            for layer in self.encoder.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
        return self

    def forward(self, inputs):
        x = (inputs - self.input_mean) / self.input_std
        skips = [self.detail(x)]
        if self.encoder_name.startswith("resnet"):
            enc = self.encoder
            x = enc.relu(enc.bn1(enc.conv1(x)))
            skips.append(x)
            x = enc.maxpool(x)
            for layer in [enc.layer1, enc.layer2, enc.layer3, enc.layer4]:
                x = layer(x)
                skips.append(x)
        elif self.encoder_name == "efficientnet_b0":
            for i, layer in enumerate(self.encoder):
                x = layer(x)
                if i in (1, 2, 3, 5, 7):
                    skips.append(x)
        else:
            skips.append(self.half_detail(F.avg_pool2d(x, 2)))
            for i, layer in enumerate(self.encoder):
                x = layer(x)
                if i % 2:
                    skips.append(x)
        for block, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], 1))
        return self.head(self.dropout(x))


class Sampler:
    def __init__(self, images, labels, cfg, device):
        self.x = torch.from_numpy(images).to(device)
        self.y = torch.from_numpy(labels).to(device)
        self.cfg, self.device = cfg, device

    def batch(self):
        cfg, device = self.cfg, self.device
        n, patch, size = cfg["batch"], cfg["patch"], cfg["size"]
        if cfg.get("augmentation") in ("d4", "flips"):
            inputs, targets = [], []
            for _ in range(n):
                index = int(np.random.randint(len(self.x)))
                top = int(np.random.randint(size - patch + 1))
                left = int(np.random.randint(size - patch + 1))
                x = self.x[index, :, top:top + patch, left:left + patch]
                y = self.y[index, :, top:top + patch, left:left + patch]
                turns = int(np.random.randint(4)) if cfg["augmentation"] == "d4" else 0
                x, y = torch.rot90(x, turns, (-2, -1)), torch.rot90(y, turns, (-2, -1))
                if np.random.random() < .5:
                    x, y = x.flip(-1), y.flip(-1)
                if cfg["augmentation"] == "flips" and np.random.random() < .5:
                    x, y = x.flip(-2), y.flip(-2)
                inputs.append(x)
                targets.append(y)
            x, y = torch.stack(inputs), torch.stack(targets)
            gamma = .9 + .2 * torch.rand(n, 1, 1, 1, device=device)
            gain = .9 + .2 * torch.rand(n, 1, 1, 1, device=device)
            x = (x.clamp_min(0).pow(gamma) * gain).clamp(0, 1)
            return x.contiguous(memory_format=torch.channels_last), y
        indices = torch.randint(len(self.x), (n,), device=device)
        angle = torch.rand(n, device=device) * 2 * math.pi
        scale = (.75 + .5 * torch.rand(n, device=device)) * patch / size
        cosine, sine = angle.cos() * scale, angle.sin() * scale
        mirror = torch.randint(2, (n,), device=device).float() * 2 - 1
        extent = (cosine.abs() + sine.abs()).clamp(max=1)
        centers = (torch.rand(n, 2, device=device) * 2 - 1) * (1 - extent[:, None])
        theta = torch.zeros(n, 2, 3, device=device)
        theta[:, 0, 0], theta[:, 0, 1] = cosine * mirror, -sine
        theta[:, 1, 0], theta[:, 1, 1] = sine * mirror, cosine
        theta[:, :, 2] = centers
        channels = self.x.shape[1]
        grid = F.affine_grid(theta, (n, channels, patch, patch), align_corners=False)
        x = F.grid_sample(self.x[indices], grid, align_corners=False, padding_mode=cfg.get("padding_mode", "reflection"))
        y = F.grid_sample(self.y[indices], grid, mode="nearest", align_corners=False)
        gamma = .8 + .4 * torch.rand(n, 1, 1, 1, device=device)
        gain = .85 + .3 * torch.rand(n, 1, 1, 1, device=device)
        x = x.clamp_min(0).pow(gamma) * gain
        x += (torch.rand(n, channels, 1, 1, device=device) - .5) * .025
        x = x.clamp(0, 1)
        return x.contiguous(memory_format=torch.channels_last), y


def loss_function(logits, target, cfg):
    logits = logits.float()
    p = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if cfg["loss"] == "focal_dice":
        pt = p * target + (1 - p) * (1 - target)
        bce = bce * (1 - pt).pow(1.5)
    dims = (1, 2, 3)
    dice = (2 * (p * target).sum(dims) + 1) / (p.sum(dims) + target.sum(dims) + 1)
    weight = float(cfg["dice_weight"])
    base = (1 - weight) * bce.mean() + weight * (1 - dice.mean())
    if cfg["loss"] == "cldice":
        from topology_loss import cldice_loss
        alpha = float(cfg.get("cldice_weight", .2))
        if not 0 <= alpha <= 1:
            raise ValueError("Invalid clDice mixing weight")
        return (1 - alpha) * base + alpha * cldice_loss(p, target, int(cfg.get("cldice_iterations", 8)))
    return base


def make_model(cfg, pretrained=True):
    if cfg["encoder"] == "vessel_local_unet":
        from vascx_features import VesselLocalUNet
        return VesselLocalUNet(cfg, pretrained)
    if cfg["encoder"] == "visionfm_hybrid":
        from visionfm_adapter import VisionFMHybrid
        return VisionFMHybrid(cfg, pretrained)
    if cfg["encoder"] == "visionfm_fundus":
        from visionfm_adapter import VisionFMUNet
        return VisionFMUNet(cfg, pretrained)
    if cfg["encoder"] == "retfound_green":
        from retfound_adapter import RETFoundUNet
        return RETFoundUNet(cfg, pretrained)
    if cfg["encoder"] == "hrnet_w18":
        from hrnet_adapter import HRNetUNet
        return HRNetUNet(cfg, pretrained)
    if cfg["encoder"] == "vascx_vessels":
        from vascx_adapter import VascXAdapter
        return VascXAdapter(cfg, pretrained)
    if cfg["encoder"] == "qtseg_fives":
        from qtseg_adapter import QTSegAdapter
        return QTSegAdapter(cfg, pretrained)
    if cfg["encoder"] == "frunet_chase":
        from frunet_adapter import FRUNetAdapter
        return FRUNetAdapter(cfg, pretrained)
    return PretrainedUNet(cfg, pretrained)


def pretraining_info(cfg):
    if cfg.get("external_init_paths"):
        if not cfg.get("external_init_metadata"):
            raise ValueError("External initialization provenance is available only after checked loading")
        return cfg["external_init_metadata"]
    if cfg["encoder"] == "vessel_local_unet":
        from vascx_features import provenance
        return provenance(int(cfg.get("vascx_member", 0)))
    if cfg["encoder"] in ("visionfm_fundus", "visionfm_hybrid"):
        from visionfm_adapter import provenance
        return provenance(cfg)
    if cfg["encoder"] == "retfound_green":
        from retfound_adapter import provenance
        return provenance(cfg)
    if cfg["encoder"] == "hrnet_w18":
        from hrnet_adapter import provenance
        return provenance()
    if cfg["encoder"] == "vascx_vessels":
        from vascx_adapter import provenance
        result = provenance()
        result.update(finetuning_preprocess=cfg["preprocess"],
                      external_member=cfg.get("vascx_member", 0))
        return result
    if cfg["encoder"] == "qtseg_fives":
        from qtseg_adapter import provenance
        return provenance()
    if cfg["encoder"] == "frunet_chase":
        from frunet_adapter import provenance
        return provenance()
    source = encoder_source(cfg["encoder"])
    return {"enum": str(source), "url": source.url,
            "dataset": "ImageNet-1K classification; no task-supervised retinal weights"}


def training_amp(cfg, device):
    return amp_context(device) if cfg.get("amp", True) else contextlib.nullcontext()


@torch.no_grad()
def ema_update(ema, model, decay):
    for destination, source in zip(ema.parameters(), model.parameters()):
        destination.lerp_(source, 1 - decay)
    for destination, source in zip(ema.buffers(), model.buffers()):
        if destination.is_floating_point():
            destination.lerp_(source, 1 - decay)
        else:
            destination.copy_(source)


@torch.inference_mode()
def predict_tiled_native(model, x, device, patch=256, overlap=.5, batch=8):
    """Match the crop scale seen in training and average overlapping interiors."""
    height, width = x.shape[-2:]
    border = 32
    padded = F.pad(x, (border, border, border, border), mode="reflect")
    ph, pw = padded.shape[-2:]
    stride = max(1, round(patch * (1 - overlap)))
    ys = np.linspace(0, ph - patch, max(1, math.ceil((ph - patch) / stride)) + 1).round().astype(int).tolist()
    xs = np.linspace(0, pw - patch, max(1, math.ceil((pw - patch) / stride)) + 1).round().astype(int).tolist()
    positions = [(y, xx) for y in ys for xx in xs]
    ramp = torch.linspace(-1, 1, patch, device=device)
    window = torch.exp(-2 * (ramp[:, None].square() + ramp[None, :].square())).clamp_min(.02)[None, None]
    total = torch.zeros(1, 1, ph, pw, device=device)
    weights = torch.zeros_like(total)
    for start in range(0, len(positions), batch):
        subset = positions[start:start + batch]
        crops = torch.cat([padded[:, :, y:y + patch, xx:xx + patch] for y, xx in subset])
        with training_amp(model.cfg, device):
            logits = model(crops.contiguous(memory_format=torch.channels_last))
        probabilities = logits.float().sigmoid()
        for probability, (y, xx) in zip(probabilities, subset):
            total[:, :, y:y + patch, xx:xx + patch] += probability[None] * window
            weights[:, :, y:y + patch, xx:xx + patch] += window
    native = total / weights.clamp_min(1e-8)
    return native[:, :, border:border + height, border:border + width]


@torch.inference_mode()
def predict(model, image, device, tta=1, geometry=None):
    model.eval()
    x = torch.from_numpy(image[None]).to(device).contiguous(memory_format=torch.channels_last)
    if tta >= 8:
        transforms = [(turns, dims) for turns in range(4) for dims in [(), (-1,)]]
    else:
        transforms = [(0, dims) for dims in ([(), (-1,), (-2,), (-2, -1)] if tta >= 4 else [()])]
    result = []
    for turns, dims in transforms:
        transformed = torch.rot90(x, turns, (-2, -1)) if turns else x
        transformed = torch.flip(transformed, dims) if dims else transformed
        if model.cfg.get("inference_mode", "full") == "tiled":
            probability = predict_tiled_native(model, transformed, device,
                patch=int(model.cfg.get("inference_patch", model.cfg["patch"])), batch=int(model.cfg.get("inference_batch", 8)))
        else:
            with training_amp(model.cfg, device):
                logits = model(transformed)
            probability = logits.float().sigmoid()
        if dims:
            probability = torch.flip(probability, dims)
        if turns:
            probability = torch.rot90(probability, -turns, (-2, -1))
        if geometry is None and probability.shape[-2:] != (512, 512):
            probability = F.interpolate(probability, size=(512, 512), mode="bilinear", align_corners=False)
        result.append(probability)
    p = torch.stack(result).mean(0)[0, 0].cpu().numpy()
    if geometry is not None:
        if geometry.get("kind") == "native_pixels":
            top, left = geometry.get("top", 0), geometry.get("left", 0)
            native = p[top:top + geometry["height"], left:left + geometry["width"]]
            p = native.astype(np.float32) if geometry.get("return_native") else np.asarray(
                Image.fromarray(native).resize((512, 512), Image.Resampling.NEAREST), dtype=np.float32)
        elif geometry.get("kind") == "resized_native_first":
            native = Image.fromarray(p).resize((geometry["width"], geometry["height"]), Image.Resampling.BILINEAR)
            p = np.asarray(native.resize((512, 512), Image.Resampling.NEAREST), dtype=np.float32)
        elif geometry.get("kind") == "crop_native_first":
            from vascx_adapter import restore_native_first
            p = restore_native_first(p, geometry)
        else:
            from vascx_adapter import restore_probability
            p = restore_probability(p, geometry)
    if not np.isfinite(p).all():
        raise FloatingPointError("Nonfinite inference probabilities")
    return p


def save_model(path, model, cfg, fold, split, epoch, score):
    torch.save({"model": model.state_dict(), "config": cfg, "fold": int(fold), "split": split,
                "epoch": epoch, "val_dice_0_5": score,
                "pretrained_source": pretraining_info(cfg)}, path)


def train_fold(args):
    cfg = config_from(args.config)
    device = setup(cfg.get("training_seed", cfg["seed"]), args.device)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    out = Path(args.out) / f"fold_{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists():
        raise FileExistsError(f"Preserve existing run: {out}")
    train_ids, val_ids = split_for(args.data, args.fold, cfg)
    fixed_final = cfg.get("checkpoint_selection") == "fixed_final_ema"
    split = {"train_ids": train_ids, "val_ids": val_ids, "seed": cfg["seed"], "fold": int(args.fold),
             "cv_protocol": cfg.get("cv_protocol", FIVEFOLD)}
    write_json(out / "split.json", split)
    write_json(out / "config.json", cfg)
    images, labels = load_data(args.data, train_ids, cfg)
    sampler = Sampler(images, labels, cfg, device)
    image_index = index_images(Path(args.data) / "train/image")
    def validation_data():
        # LOO calls this only after the fixed training budget and checkpoint save.
        val_images = np.stack([cv_preprocess(image_index[key], cfg) for key in val_ids])
        val_targets = read_targets(args.data, val_ids, cfg.get("evaluation_protocol", LEGACY_PROTOCOL))
        return val_images, val_targets, np.stack([fov512(image_index[key]) for key in val_ids]), prediction_geometries(image_index, val_ids, cfg)
    if not fixed_final:
        val_x, targets, fovs, geometries = validation_data()
    model = make_model(cfg, pretrained=not bool(cfg.get("external_init_paths")))
    if cfg.get("external_init_paths"):
        from external_pretraining import load_initialization
        cfg["external_init_metadata"] = load_initialization(model, cfg, args.fold)
        write_json(out / "config.json", cfg)
    model = model.to(device, memory_format=torch.channels_last)
    if cfg["encoder"] == "vascx_vessels":
        # deepcopy of this upstream ScriptModule produces non-leaf parameters.
        # Recreate its graph, then copy tensors into an independent EMA model.
        ema = make_model(cfg, pretrained=False).to(device, memory_format=torch.channels_last)
        ema.load_state_dict(model.state_dict(), strict=True)
        ema.eval().requires_grad_(False)
    else:
        ema = copy.deepcopy(model).eval().requires_grad_(False)
    encoder_params = model.pretrained_parameters() if hasattr(model, "pretrained_parameters") else list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW([{"params": encoder_params, "lr": cfg["lr"] * cfg["encoder_lr_factor"]},
                                   {"params": decoder_params, "lr": cfg["lr"]}], weight_decay=cfg["weight_decay"])
    def schedule(epoch):
        warmup = min(1., (epoch + 1) / 5)
        return warmup * (.05 + .95 * .5 * (1 + math.cos(math.pi * epoch / cfg["epochs"])))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.get("amp", True))
    best, best_epoch, start = -1., 0, time.monotonic()
    print(f"FOLD_START {cfg['name']} fold={args.fold} device={device} train={train_ids} val={val_ids}", flush=True)
    with (out / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "loss", "val_dice_0_5", "seconds"])
        writer.writeheader()
        for epoch in range(1, cfg["epochs"] + 1):
            epoch_start = time.monotonic()
            model.train()
            total = 0.
            for _ in range(cfg["steps"]):
                x, y = sampler.batch()
                optimizer.zero_grad(set_to_none=True)
                with training_amp(cfg, device):
                    loss = loss_function(model(x), y, cfg)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                total += float(loss.detach())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                if not torch.isfinite(norm):
                    if not scaler.is_enabled():
                        raise FloatingPointError("Nonfinite gradient in full-precision training")
                    # AMP can overflow during scale discovery; skip this step,
                    # let GradScaler lower the scale, and do not update EMA.
                    scaler.step(optimizer)
                    scaler.update()
                    continue
                scaler.step(optimizer)
                scaler.update()
                ema_update(ema, model, cfg["ema_decay"])
            scheduler.step()
            score = None
            if not fixed_final and (epoch == 1 or epoch % cfg["validate_every"] == 0 or epoch == cfg["epochs"]):
                p = np.stack([predict(ema, x, device, 1, geometry) * fov
                              for x, fov, geometry in zip(val_x, fovs, geometries)])
                score = float(dice_per_image(p, targets, .5).mean())
                if score > best:
                    best, best_epoch = score, epoch
                    save_model(out / "best.pt", ema, cfg, args.fold, split, epoch, score)
            if fixed_final and epoch == cfg["epochs"]:
                best_epoch = epoch
                save_model(out / "best.pt", ema, cfg, args.fold, split, epoch, None)
                write_json(out / "selection_witness.json", {"checkpoint_selection": "fixed_final_ema",
                    "training_epochs": epoch, "steps_per_epoch": cfg["steps"], "heldout_labels_opened_during_training": False,
                    "heldout_id": val_ids, "cv_protocol": cfg.get("cv_protocol", FIVEFOLD)})
            duration = time.monotonic() - epoch_start
            writer.writerow({"epoch": epoch, "loss": total / cfg["steps"], "val_dice_0_5": score, "seconds": duration})
            stream.flush()
            progress = {"epoch": epoch, "epochs": cfg["epochs"], "best_epoch": best_epoch,
                        "best_val_dice_0_5": best, "elapsed_seconds": time.monotonic() - start}
            write_json(out / "progress.json", progress)
            print(f"epoch={epoch} loss={total/cfg['steps']:.5f} val={score} best={best:.6f}@{best_epoch} sec={duration:.1f}", flush=True)
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    ema.load_state_dict(checkpoint["model"])
    if fixed_final:
        val_x, targets, fovs, geometries = validation_data()
    probabilities = np.stack([predict(ema, image, device, cfg["tta"], geometry) * fov
                              for image, fov, geometry in zip(val_x, fovs, geometries)])
    np.savez_compressed(out / "oof.npz", ids=np.array(val_ids), probabilities=probabilities.astype(np.float32), targets=targets)
    threshold, sweep = (.5, []) if fixed_final else choose_threshold(probabilities, targets)
    report = {"name": cfg["name"], "split": split, "best_epoch": best_epoch, "threshold": threshold,
              "mean_vessel_dice_0_5": float(dice_per_image(probabilities, targets, .5).mean()),
              "mean_vessel_dice": float(dice_per_image(probabilities, targets, threshold).mean()),
              "threshold_sweep": sweep, "seconds": time.monotonic() - start,
              "pretrained_source": checkpoint["pretrained_source"], "config": cfg,
              "evaluation_protocol": cfg.get("evaluation_protocol", LEGACY_PROTOCOL),
              "cv_protocol": cfg.get("cv_protocol", FIVEFOLD),
              "checkpoint_selection": cfg.get("checkpoint_selection", "validation_best")}
    write_json(out / "report.json", report)
    print(f"FOLD_COMPLETE {args.fold} dice={report['mean_vessel_dice']:.6f}", flush=True)


def finalize(root, out):
    out = Path(out)
    ids = [p.stem for p in image_paths(Path(root) / "train/image")]
    records, reports = {}, []
    first = json.loads((out / "fold_0/report.json").read_text(encoding="utf-8"))
    cv_protocol = first["config"].get("cv_protocol", FIVEFOLD)
    evaluation_protocol = first["config"].get("evaluation_protocol", LEGACY_PROTOCOL)
    for fold in range(20 if cv_protocol == LOO else 5):
        directory = out / f"fold_{fold}"
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if (report["config"].get("cv_protocol", FIVEFOLD) != cv_protocol
                or report["config"].get("evaluation_protocol", LEGACY_PROTOCOL) != evaluation_protocol):
            raise ValueError("Cannot combine folds from different evaluation protocols")
        reports.append(report)
        split = report["split"]
        train_ids, val_ids = split_for(root, fold, first["config"])
        if split["train_ids"] != train_ids or split["val_ids"] != val_ids:
            raise ValueError("Fold differs from the declared image-level protocol")
        if cv_protocol == LOO:
            witness = json.loads((directory / "selection_witness.json").read_text(encoding="utf-8"))
            if (report.get("checkpoint_selection") != "fixed_final_ema" or report["best_epoch"] != first["config"]["epochs"]
                    or witness["heldout_labels_opened_during_training"] is not False):
                raise ValueError("LOO checkpoint was not fixed independently of its heldout label")
        if set(split["train_ids"]) & set(split["val_ids"]):
            raise ValueError("Training/validation leakage")
        with np.load(directory / "oof.npz", allow_pickle=False) as saved:
            if saved["ids"].tolist() != split["val_ids"] or saved["probabilities"].dtype != np.float32:
                raise ValueError("Invalid OOF archive")
            for key, probability, target in zip(saved["ids"], saved["probabilities"], saved["targets"]):
                if key in records or not np.isfinite(probability).all():
                    raise ValueError("Duplicated or invalid OOF prediction")
                records[str(key)] = (probability, target)
    if len(records) != 20 or set(records) != set(ids):
        raise ValueError("OOF must cover all 20 training images exactly once")
    p = np.stack([records[key][0] for key in ids])
    y = np.stack([records[key][1] for key in ids])
    expected = read_targets(root, ids, evaluation_protocol)
    if not np.array_equal(y, expected):
        raise ValueError("OOF targets differ from original canonical labels")
    threshold, sweep = choose_threshold(p, y)
    scores, fixed = dice_per_image(p, y, threshold), dice_per_image(p, y, .5)
    report = {"name": reports[0]["name"], "n_images": len(ids), "threshold": threshold,
              "mean_vessel_dice": float(scores.mean()), "mean_vessel_dice_0_5": float(fixed.mean()),
              "per_image": [{"id": key, "vessel_dice": float(a), "vessel_dice_0_5": float(b)} for key, a, b in zip(ids, scores, fixed)],
              "folds": [{"fold": i, "split": r["split"], "best_epoch": r["best_epoch"],
                         "mean_vessel_dice_0_5": r["mean_vessel_dice_0_5"],
                         "pretrained_source": r["pretrained_source"]} for i, r in enumerate(reports)],
              "config": reports[0]["config"], "pretrained_source": reports[0]["pretrained_source"],
              "threshold_sweep": sweep,
              "evaluation_protocol": evaluation_protocol, "cv_protocol": cv_protocol,
              "checkpoint_selection": first["config"].get("checkpoint_selection", "validation_best"),
              "metric": "all 20 images, full-image macro vessel Dice; target protocol explicitly recorded",
              "selection_note": ("Fixed training epochs and final EMA; heldout labels read only after training. Global threshold selected on full20 OOF; tuning validation, not independent test."
                  if cv_protocol == LOO else "OOF labels choose checkpoints/threshold and configurations; tuning validation, not independent test. No competition submission.")}
    np.savez_compressed(out / "oof_predictions.npz", ids=np.array(ids), probabilities=p.astype(np.float32), targets=y)
    write_json(out / "report.json", report)
    print("OOF_COMPLETE " + json.dumps({k: report[k] for k in ["name", "n_images", "threshold", "mean_vessel_dice", "mean_vessel_dice_0_5"]}), flush=True)
    return report


def run_all(args):
    cfg = config_from(args.config)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "run_config.json"
    write_json(cfg_path, cfg)
    cv2.setNumThreads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for Kaggle training")
    count = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(min(count, 2))]
    witness = {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
               "opencv": cv2.__version__, "numpy": np.__version__, "devices": []}
    if cfg["encoder"] in ("hrnet_w18", "retfound_green"):
        import timm
        witness["timm"] = timm.__version__
    for device in devices:
        torch.manual_seed(42)
        matrix = torch.randn(32, 32, device=device)
        output = matrix @ matrix
        assert torch.isfinite(output).all() and output.abs().sum() > 0
        witness["devices"].append({"device": device, "name": torch.cuda.get_device_name(device), "kernel_witness": "passed"})
    write_json(out / "environment_witness.json", witness)
    print("GPU_WITNESS " + json.dumps(witness), flush=True)
    # Download classification weights once before GPU workers share the cache.
    if cfg.get("external_init_paths"):
        if len(cfg["external_init_paths"]) != 2 or not all(Path(path).is_file() for path in cfg["external_init_paths"]):
            raise ValueError("Both external initialization files must be mounted before launching folds")
    elif cfg["encoder"] in ("visionfm_fundus", "visionfm_hybrid"):
        from visionfm_adapter import ensure_assets
        ensure_assets()
        if cfg["encoder"] == "visionfm_hybrid":
            source_weights = models.ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
            del source_weights
    elif cfg["encoder"] == "retfound_green":
        from retfound_adapter import ensure_assets
        ensure_assets()
        if cfg.get("cnn_skip") == "resnet18":
            source_weights = models.ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
            del source_weights
        elif cfg.get("cnn_skip") == "vascx":
            from vascx_adapter import ensure_assets as ensure_vessel_assets
            ensure_vessel_assets()
    elif cfg["encoder"] == "hrnet_w18":
        from hrnet_adapter import ensure_assets
        ensure_assets()
    elif cfg["encoder"] == "qtseg_fives":
        from qtseg_adapter import ensure_assets
        ensure_assets()
    elif cfg["encoder"] in ("vascx_vessels", "vessel_local_unet"):
        from vascx_adapter import ensure_assets
        ensure_assets()
    elif cfg["encoder"] == "frunet_chase":
        from frunet_adapter import ensure_assets
        ensure_assets()
    else:
        source_weights = encoder_source(cfg["encoder"]).get_state_dict(progress=True, check_hash=True)
        del source_weights
    gc.collect()
    pending, running = list(range(20 if cfg.get("cv_protocol") == LOO else 5)), {}
    last_log = 0.
    while pending or running:
        used = {value[1] for value in running.values()}
        for device in devices:
            if pending and device not in used:
                fold = pending.pop(0)
                directory = out / f"fold_{fold}"
                directory.mkdir(exist_ok=True)
                stream = (directory / "worker.log").open("w", encoding="utf-8")
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "train", "--config", str(cfg_path),
                           "--data", str(Path(args.data).resolve()), "--out", str(out), "--fold", str(fold), "--device", device]
                env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
                running[fold] = (process, device, stream)
                print(f"LAUNCH fold={fold} device={device} pid={process.pid}", flush=True)
        for fold, (process, device, stream) in list(running.items()):
            result = process.poll()
            if result is None:
                continue
            stream.close()
            del running[fold]
            if result:
                for other, _, other_stream in running.values():
                    other.terminate()
                    other.wait()
                    other_stream.close()
                print((out / f"fold_{fold}" / "worker.log").read_text(encoding="utf-8")[-16000:], flush=True)
                raise RuntimeError(f"Fold {fold} failed, exit={result}")
            print(f"FINISHED fold={fold} device={device}", flush=True)
        if time.monotonic() - last_log >= 30:
            status = {}
            for fold in running:
                progress = out / f"fold_{fold}" / "progress.json"
                if progress.exists():
                    try:
                        status[str(fold)] = json.loads(progress.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pass
            print("PROGRESS " + json.dumps(status), flush=True)
            last_log = time.monotonic()
        if running:
            time.sleep(2)
    finalize(args.data, out)


def witness(args):
    cfg = dict(DEFAULT, encoder=args.encoder, size=512, patch=224, batch=4)
    device = setup(42, args.device)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    ids, _ = get_split(args.data, "0", 42)
    images, labels = load_data(args.data, ids[:2], cfg)
    sampler = Sampler(images, labels, cfg, device)
    model = make_model(cfg).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    times = []
    losses = []
    initial_head = model.head.weight.detach().clone()
    for _ in range(4):
        t0 = time.monotonic()
        model.train()
        x, y = sampler.batch()
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            loss = loss_function(model(x), y, cfg)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.monotonic() - t0)
        losses.append(float(loss.detach()))
    prediction = predict(model, images[0], device, 1)
    assert prediction.shape == (512, 512)
    assert not torch.equal(initial_head, model.head.weight.detach()), "Optimizer did not update model"
    report = {"encoder": args.encoder, "device": str(device), "losses": losses,
              "shape": list(prediction.shape), "parameters": sum(p.numel() for p in model.parameters()),
              "seconds_per_step": float(np.mean(times[1:])), "pretrained_source": pretraining_info(cfg),
              "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}
    write_json(Path("optimization") / f"witness_{args.encoder}.json", report)
    print("STRONG_WITNESS_OK " + json.dumps(report), flush=True)


def evaluate(args):
    device = setup(42, args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = dict(checkpoint["config"])
    cfg["inference_mode"] = args.inference_mode
    cfg["inference_patch"] = args.inference_patch
    cfg["inference_projection"] = args.projection
    model = make_model(cfg, pretrained=False).to(device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    ids = checkpoint["split"]["val_ids"]
    images, _ = load_data(args.data, ids, cfg)
    image_index = index_images(Path(args.data) / "train/image")
    label_index = index_images(Path(args.data) / "train/label")
    targets = read_targets(args.data, ids, cfg.get("evaluation_protocol", LEGACY_PROTOCOL))
    start = time.monotonic()
    geometries = prediction_geometries(image_index, ids, cfg)
    probabilities = np.stack([predict(model, x, device, args.tta, geometry) * fov512(image_index[key])
                              for key, x, geometry in zip(ids, images, geometries)])
    threshold, sweep = choose_threshold(probabilities, targets)
    report = {"checkpoint": args.checkpoint, "split": checkpoint["split"], "config": cfg,
              "inference_mode": args.inference_mode, "inference_patch": args.inference_patch,
              "n_images": len(ids), "threshold": threshold,
              "mean_vessel_dice": float(dice_per_image(probabilities, targets, threshold).mean()),
              "mean_vessel_dice_0_5": float(dice_per_image(probabilities, targets, .5).mean()),
              "per_image": dict(zip(ids, dice_per_image(probabilities, targets, threshold).tolist())),
              "seconds": time.monotonic() - start, "threshold_sweep": sweep,
              "scope": "partial-fold tuning validation; never a complete OOF result by itself"}
    out = Path(args.out)
    write_json(out / "report.json", report)
    np.savez_compressed(out / "oof.npz", ids=np.array(ids), probabilities=probabilities.astype(np.float32), targets=targets)
    print(json.dumps({key: report[key] for key in report if key != "threshold_sweep"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["train", "run", "finalize", "witness", "evaluate"]:
        sub = commands.add_parser(name)
        sub.add_argument("--data", default="data/segmentation")
        sub.add_argument("--out", default="optimization/local_strong")
        if name in ["train", "run"]:
            sub.add_argument("--config", required=True)
        if name in ["train", "witness", "evaluate"]:
            sub.add_argument("--device", default="auto")
        if name == "train":
            sub.add_argument("--fold", required=True, choices=[str(i) for i in range(20)])
        if name == "witness":
            sub.add_argument("--encoder", default="resnet34", choices=["resnet34", "resnet18", "convnext_tiny", "efficientnet_b0"])
        if name == "evaluate":
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--inference-mode", choices=["full", "tiled"], default="tiled")
            sub.add_argument("--inference-patch", type=int, default=256)
            sub.add_argument("--projection", choices=["direct", "native_first"], default="direct")
            sub.add_argument("--tta", type=int, choices=[1, 4, 8], default=4)
    args = parser.parse_args()
    if args.command == "train":
        train_fold(args)
    elif args.command == "run":
        run_all(args)
    elif args.command == "finalize":
        finalize(args.data, args.out)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        witness(args)


if __name__ == "__main__":
    main()
